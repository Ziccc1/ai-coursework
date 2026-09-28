from __future__ import annotations

import csv
import json
import re
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, hinge_loss, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
TRAIN_PATH = ROOT / "train_data.csv"
TEST_PATH = ROOT / "test_data_unlabeled.csv"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
CV_FOLDS = 3
MAX_FEATURES = 20_000
MIN_DF = 2
SUBLINEAR_TF = True
BASE_NGRAM_RANGE = (1, 2)
CURVE_EPOCHS = 200
CORE_MODEL_NAMES = ("MultinomialNB", "SVC", "LogisticRegression", "MLPClassifier")
SGD_MODEL_LOSSES = {
    "SGDClassifier_log_loss": "log_loss",
    "SGDClassifier_hinge": "hinge",
}
MODEL_DISPLAY_NAMES = {
    "MultinomialNB": "朴素贝叶斯",
    "SVC": "支持向量机 SVC",
    "LogisticRegression": "逻辑回归",
    "MLPClassifier": "多层感知机 MLP",
    "SGDClassifier_log_loss": "SGD 逻辑分类器",
    "SGDClassifier_hinge": "SGD 线性 SVM",
}

# 检查邮件头中是否存在直接表示类别的字段，避免明显的标签泄露。
LABEL_HEADER_NAMES = {"newsgroup", "newsgroups", "target", "category", "label", "class"}


def load_labeled_data() -> tuple[list[str], np.ndarray]:
    """只读取有标签数据；实验阶段不读取无标签测试集。"""
    csv.field_size_limit(2**31 - 1)
    train_df = pd.read_csv(TRAIN_PATH, encoding="utf-8", engine="python")
    required = {"text", "target"}
    if not required.issubset(train_df.columns):
        raise ValueError(f"{TRAIN_PATH.name} must contain columns {sorted(required)}")
    return train_df["text"].fillna("").astype(str).tolist(), train_df["target"].to_numpy()


def load_unlabeled_test() -> list[str]:
    """最终生成预测时才读取无标签测试集。"""
    test_df = pd.read_csv(TEST_PATH, encoding="utf-8", engine="python")
    if "text" not in test_df.columns:
        raise ValueError(f"{TEST_PATH.name} must contain a 'text' column")
    return test_df["text"].fillna("").astype(str).tolist()


def print_labeled_data_preview(texts: list[str], labels: np.ndarray) -> None:
    """有标签数据的加载结果。"""
    print("--- 数据加载成功 ---", flush=True)
    print(f"训练集样本数量: {len(texts)}", flush=True)
    print(f"训练集标签数量: {len(labels)}", flush=True)
    print("-" * 20, flush=True)

    if texts:
        print("第一个训练样本内容:", flush=True)
        print(texts[0], flush=True)
        print(f"\n第一个训练样本的标签: {labels[0]}", flush=True)
        print("-" * 20, flush=True)


def configure_chinese_plots() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    available = [name for name in candidates if name in installed]
    plt.rcParams["font.sans-serif"] = available + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

#数据清洗消融实验的设置
def header_keys(document: str) -> set[str]:
    """提取首个空行之前的邮件头字段名。"""
    boundary = re.search(r"\r?\n\r?\n", document)
    if boundary is None:
        return set()
    keys: set[str] = set()
    for line in document[: boundary.start()].splitlines():
        if ":" in line and not line[:1].isspace():
            keys.add(line.split(":", 1)[0].strip().casefold())
    return keys


def audit_target_headers(documents: list[str]) -> None:
    found = sorted({key for doc in documents for key in header_keys(doc) if key in LABEL_HEADER_NAMES})
    if found:
        raise ValueError(
            "Potential direct label fields were found in the message headers: "
            f"{found}. Remove those fields uniformly from every condition before fitting."
        )


def split_header_body(document: str) -> tuple[str | None, str]:
    boundary = re.search(r"\r?\n\r?\n", document)
    if boundary is None:
        return None, document
    return document[: boundary.start()], document[boundary.end() :]


def subject_only(document: str) -> str:
    """删除非 Subject 邮件头，保留 Subject 和正文。"""
    header, body = split_header_body(document)
    if header is None:
        return document

    kept_subject: list[str] = []
    collecting_subject = False
    for line in header.splitlines():
        if line[:1].isspace():
            if collecting_subject:
                kept_subject.append(line)
            continue
        collecting_subject = False
        if ":" in line and line.split(":", 1)[0].strip().casefold() == "subject":
            kept_subject.append(line)
            collecting_subject = True

    if kept_subject:
        return "\n".join([*kept_subject, "", body])
    return body


def body_only(document: str) -> str:
    """删除整个邮件头；没有邮件头的文档保持原样。"""
    header, body = split_header_body(document)
    return document if header is None else body


def remove_quoted_lines(document: str) -> str:
    return "\n".join(line for line in document.splitlines() if not re.match(r"^\s*>", line))


def remove_signature(document: str) -> str:
    lines = document.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^\s*--\s*$", line):
            return "\n".join(lines[:index])
    return document


def clean_documents(documents: list[str], condition: str) -> list[str]:
    if condition == "raw":
        return list(documents)
    if condition == "keep_subject":
        return [subject_only(doc) for doc in documents]
    if condition == "body_only":
        return [body_only(doc) for doc in documents]
    if condition == "without_quotes":
        return [remove_quoted_lines(subject_only(doc)) for doc in documents]
    if condition == "without_signatures":
        return [remove_signature(subject_only(doc)) for doc in documents]
    raise ValueError(f"Unknown cleaning condition: {condition}")


def make_pipeline(classifier, *, analyzer: str = "word", ngram_range: tuple[int, int] = BASE_NGRAM_RANGE) -> Pipeline:
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        sublinear_tf=SUBLINEAR_TF,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])

#根据TUNNING确定的模型参数搜索范围
def model_specs() -> dict[str, tuple[object, dict[str, list]]]:
    specs = {
        "MultinomialNB": (
            MultinomialNB(),
            {"clf__alpha": [0.1, 1.0, 10.0]},
        ),
        "SVC": (
            SVC(kernel="linear", cache_size=1000),
            {"clf__C": [0.1, 1.0, 10.0, 100.0]},
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"clf__C": [0.1, 1.0, 10.0, 100.0]},
        ),
        "MLPClassifier": (
            MLPClassifier(
                max_iter=200,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
            {
                "clf__hidden_layer_sizes": [(50,), (100,)],
                "clf__alpha": [0.0001, 0.00001],
            },
        ),
    }
    for name, loss in SGD_MODEL_LOSSES.items():
        specs[name] = (
            SGDClassifier(
                loss=loss,
                max_iter=200,
                tol=None,
                random_state=RANDOM_STATE,
                learning_rate="optimal",
                shuffle=True,
            ),
            {"clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2]},
        )
    return specs


def classifier_of(estimator):
    return estimator.named_steps["clf"] if isinstance(estimator, Pipeline) else estimator


def evaluate(estimator, features, labels) -> dict[str, float | str]:
    predicted = estimator.predict(features)
    classifier = classifier_of(estimator)
    result: dict[str, float | str] = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
    }
    if hasattr(estimator, "predict_proba"):
        result["loss_name"] = "log_loss"
        result["loss"] = float(
            log_loss(labels, estimator.predict_proba(features), labels=classifier.classes_)
        )
    elif hasattr(estimator, "decision_function"):
        result["loss_name"] = "multiclass_hinge_loss_from_decision_scores"
        result["loss"] = float(
            hinge_loss(labels, estimator.decision_function(features), labels=classifier.classes_)
        )
    else:
        result["loss_name"] = "unavailable"
        result["loss"] = float("nan")
    return result


def dump_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_main_model_comparison(X_train, y_train, X_val, y_val, cv, out_dir: Path):
    searches = {}
    rows = []
    cv_dir = out_dir / "cv_results"
    model_dir = out_dir / "models"
    cv_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, (classifier, grid) in model_specs().items():
        combinations = int(np.prod([len(values) for values in grid.values()]))
        print(
            f"\n[模型搜索] {name}：{combinations} 组参数，"
            f"三折共 {combinations * CV_FOLDS} 次拟合。",
            flush=True,
        )

        search = GridSearchCV(
            estimator=make_pipeline(classifier),
            param_grid=grid,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            refit="accuracy",
            cv=cv,
            n_jobs=1,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)

        print(f"[模型搜索] {name} 的所有交叉验证拟合已完成。", flush=True)
        for index, params in enumerate(search.cv_results_["params"]):
            cv_accuracy = search.cv_results_["mean_test_accuracy"][index]
            cv_accuracy_std = search.cv_results_["std_test_accuracy"][index]
            cv_f1 = search.cv_results_["mean_test_macro_f1"][index]
            print(
                f"  参数 {params}："
                f"CV Accuracy={cv_accuracy:.4f}±{cv_accuracy_std:.4f}，"
                f"CV Macro-F1={cv_f1:.4f}",
                flush=True,
            )

        best = search.best_estimator_
        train_metrics = evaluate(best, X_train, y_train)
        val_metrics = evaluate(best, X_val, y_val)
        best_index = int(search.best_index_)
        cv_std = float(search.cv_results_["std_test_accuracy"][best_index])

        rows.append(
            {
                "model": name,
                "best_params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
                "cv_accuracy_mean": float(search.best_score_),
                "cv_accuracy_std": cv_std,
                "cv_macro_f1_mean": float(search.cv_results_["mean_test_macro_f1"][best_index]),
                "cv_macro_f1_std": float(search.cv_results_["std_test_macro_f1"][best_index]),
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_loss_name": train_metrics["loss_name"],
                "train_loss": train_metrics["loss"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "validation_loss_name": val_metrics["loss_name"],
                "validation_loss": val_metrics["loss"],
                "n_features": len(best.named_steps["tfidf"].vocabulary_),
            }
        )
        pd.DataFrame(search.cv_results_).to_csv(
            cv_dir / f"{name}_grid.csv",
            index=False,
            encoding="utf-8-sig",
        )
        joblib.dump(best, model_dir / f"{name}_best_on_outer_train.joblib")
        searches[name] = search

        print(
            f"[模型完成] {name}："
            f"CV Accuracy={search.best_score_:.4f}±{cv_std:.4f}；"
            f"训练集 Accuracy={train_metrics['accuracy']:.4f}；"
            f"验证集 Accuracy={val_metrics['accuracy']:.4f}；"
            f"验证集 Macro-F1={val_metrics['macro_f1']:.4f}；"
            f"最优参数={search.best_params_}",
            flush=True,
        )

    results = pd.DataFrame(rows).sort_values(
        ["validation_accuracy", "validation_macro_f1"], ascending=False
    )
    results.to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    return searches, results

#模型对比图片生成
def save_main_comparison_plot(results: pd.DataFrame, out_dir: Path) -> None:
    names = [MODEL_DISPLAY_NAMES[name] for name in results["model"]]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax.bar(x - width / 2, results["validation_accuracy"], width, label="验证集准确率")
    ax.bar(x + width / 2, results["validation_macro_f1"], width, label="验证集 Macro-F1")
    ax.set_title("所有模型验证集表现")
    ax.set_ylabel("分数")
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_dir / "model_comparison.png", dpi=180)
    plt.close(fig)


def save_grid_search_plots(searches, out_dir: Path) -> None:
    parameter_labels = {
        "MultinomialNB": "朴素贝叶斯：alpha",
        "SVC": "SVC：C",
        "LogisticRegression": "逻辑回归：C",
        "MLPClassifier": "MLP：隐藏层宽度与 alpha",
        "SGDClassifier_log_loss": "SGD 逻辑分类器：alpha",
        "SGDClassifier_hinge": "SGD 线性 SVM：alpha",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes = axes.ravel()
    for ax, (name, search) in zip(axes, searches.items()):
        results = pd.DataFrame(search.cv_results_)
        if name == "MLPClassifier":
            for alpha in sorted(results["param_clf__alpha"].astype(float).unique()):
                part = results[results["param_clf__alpha"].astype(float) == alpha].copy()
                widths = part["param_clf__hidden_layer_sizes"].map(lambda value: value[0])
                order = np.argsort(widths.to_numpy(dtype=float))
                ax.plot(
                    widths.to_numpy(dtype=float)[order],
                    part["mean_test_accuracy"].to_numpy(dtype=float)[order],
                    marker="o",
                    label=f"alpha={alpha:g}",
                )
            ax.set_xlabel("隐藏层神经元数")
            ax.legend(title="正则化强度")
        else:
            parameter = (
                "param_clf__alpha"
                if name == "MultinomialNB" or name in SGD_MODEL_LOSSES
                else "param_clf__C"
            )
            x_values = results[parameter].astype(float).to_numpy()
            order = np.argsort(x_values)
            ax.plot(
                x_values[order],
                results["mean_test_accuracy"].to_numpy(dtype=float)[order],
                marker="o",
            )
            ax.set_xscale("log")
            ax.set_xlabel(
                "alpha（对数刻度）"
                if name == "MultinomialNB" or name in SGD_MODEL_LOSSES
                else "C（对数刻度）"
            )
        ax.set_title(parameter_labels[name])
        ax.set_ylabel("三折 CV 平均准确率")
        ax.grid(alpha=0.25)
    fig.suptitle("GridSearchCV 参数对比（训练部分内的三折交叉验证）", fontsize=13)
    fig.savefig(out_dir / "grid_search_comparison.png", dpi=180)
    plt.close(fig)


def save_confusion_matrices(searches, X_val, y_val, out_dir: Path) -> None:
    """分别保存六个模型的验证集混淆矩阵 CSV 和图片。"""
    matrix_dir = out_dir / "confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    for name, search in searches.items():
        estimator = search.best_estimator_
        classifier = classifier_of(estimator)
        class_labels = classifier.classes_
        predicted = estimator.predict(X_val)
        matrix = confusion_matrix(y_val, predicted, labels=class_labels)
        matrix_df = pd.DataFrame(
            matrix,
            index=pd.Index(class_labels, name="真实类别"),
            columns=pd.Index(class_labels, name="预测类别"),
        )
        matrix_df.to_csv(
            matrix_dir / f"{name}_validation.csv",
            encoding="utf-8-sig",
        )

        fig, ax = plt.subplots(figsize=(9, 7.5), constrained_layout=True)
        image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(image, ax=ax, label="样本数")
        class_names = [str(value) for value in class_labels]
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="预测类别",
            ylabel="真实类别",
            title=f"{MODEL_DISPLAY_NAMES[name]}：验证集混淆矩阵",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        threshold = matrix.max() / 2 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                    fontsize=8,
                )
        fig.savefig(matrix_dir / f"{name}_validation.png", dpi=180)
        plt.close(fig)
        print(f"[混淆矩阵] {name}：CSV 和图片已保存。", flush=True)


def collect_epoch_curve(
    name: str,
    estimator,
    X_train_vec,
    y_train,
    X_val_vec,
    y_val,
    epochs: int,
    batch_size: int = 200,
):
    """记录训练块波动，以及每轮完整训练集和验证集上的指标。"""
    classes = np.unique(y_train)
    rows = []
    rng = np.random.default_rng(RANDOM_STATE)
    samples_per_epoch = len(y_train)
    batches_per_epoch = int(np.ceil(samples_per_epoch / batch_size))
    step = 0
    first_update = True

    for epoch in range(1, epochs + 1):
        shuffled_indices = rng.permutation(samples_per_epoch)
        for batch_number, start in enumerate(
            range(0, samples_per_epoch, batch_size), start=1
        ):
            batch_indices = shuffled_indices[start : start + batch_size]
            X_batch = X_train_vec[batch_indices]
            y_batch = y_train[batch_indices]
            if first_update:
                estimator.partial_fit(X_batch, y_batch, classes=classes)
                first_update = False
            else:
                estimator.partial_fit(X_batch, y_batch)

            batch_metrics = evaluate(estimator, X_batch, y_batch)
            step += 1
            rows.append(
                {
                    "experiment": name,
                    "granularity": "batch",
                    "epoch": epoch,
                    "batch": batch_number,
                    "step": step,
                    "x_epoch": epoch - 1 + batch_number / batches_per_epoch,
                    "train_accuracy": batch_metrics["accuracy"],
                    "validation_accuracy": np.nan,
                    "train_loss_name": batch_metrics["loss_name"],
                    "train_loss": batch_metrics["loss"],
                    "validation_loss_name": "",
                    "validation_loss": np.nan,
                }
            )

        train_metrics = evaluate(estimator, X_train_vec, y_train)
        val_metrics = evaluate(estimator, X_val_vec, y_val)
        rows.append(
            {
                "experiment": name,
                "granularity": "epoch",
                "epoch": epoch,
                "batch": batches_per_epoch,
                "step": step,
                "x_epoch": epoch,
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": val_metrics["accuracy"],
                "train_loss_name": train_metrics["loss_name"],
                "train_loss": train_metrics["loss"],
                "validation_loss_name": val_metrics["loss_name"],
                "validation_loss": val_metrics["loss"],
            }
        )

        print(
            f"[逐轮训练] {name} | 第 {epoch:3d}/{epochs} 轮 | "
            f"训练 Acc={train_metrics['accuracy']:.4f}，"
            f"验证 Acc={val_metrics['accuracy']:.4f} | "
            f"训练 Loss={train_metrics['loss']:.4f}，"
            f"验证 Loss={val_metrics['loss']:.4f}",
            flush=True,
        )

    return rows


def save_epoch_curves(searches, X_train, y_train, X_val, y_val, out_dir: Path):
    """记录所选 SGD 与 MLP 配置的逐轮曲线和训练块波动。"""
    first_search = searches["LogisticRegression"]
    vectorizer = first_search.best_estimator_.named_steps["tfidf"]
    X_train_vec = vectorizer.transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    curves = []
    sgd_curve_specs = [
        ("SGD logistic (log_loss)", "SGDClassifier_log_loss", "log_loss"),
        ("SGD linear SVM (hinge)", "SGDClassifier_hinge", "hinge"),
    ]
    for label, search_name, loss in sgd_curve_specs:
        search = searches[search_name]
        alpha = search.best_params_["clf__alpha"]
        curve_estimator = SGDClassifier(
            loss=loss,
            alpha=alpha,
            max_iter=1,
            tol=None,
            random_state=RANDOM_STATE,
            learning_rate="optimal",
            shuffle=True,
        )
        print(
            f"\n[曲线实验] 开始 {label}，alpha={alpha:g}，"
            f"共 {CURVE_EPOCHS} 轮。",
            flush=True,
        )
        curves.extend(
            collect_epoch_curve(
                label, curve_estimator, X_train_vec, y_train, X_val_vec, y_val, CURVE_EPOCHS
            )
        )
        pd.DataFrame(search.cv_results_).to_csv(
            out_dir / f"{label.lower().replace(' ', '_').replace('(', '').replace(')', '')}_grid.csv",
            index=False,
            encoding="utf-8-sig",
        )

    best_mlp_params = searches["MLPClassifier"].best_params_
    mlp = MLPClassifier(
        hidden_layer_sizes=best_mlp_params["clf__hidden_layer_sizes"],
        alpha=best_mlp_params["clf__alpha"],
        solver="adam",
        max_iter=1,
        early_stopping=False,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    print(
        "\n[曲线实验] 开始 MLPClassifier (Adam)，"
        f"隐藏层={best_mlp_params['clf__hidden_layer_sizes']}，"
        f"alpha={best_mlp_params['clf__alpha']:g}，"
        f"共 {CURVE_EPOCHS} 轮。",
        flush=True,
    )
    curves.extend(
        collect_epoch_curve(
            "MLPClassifier (Adam)", mlp, X_train_vec, y_train, X_val_vec, y_val, CURVE_EPOCHS
        )
    )

    curve_df = pd.DataFrame(curves)
    curve_df.to_csv(out_dir / "epoch_curves.csv", index=False, encoding="utf-8-sig")
    print(
        f"[曲线实验] 逐轮和训练块数据已保存：{out_dir / 'epoch_curves.csv'}",
        flush=True,
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    order = ["SGD logistic (log_loss)", "SGD linear SVM (hinge)", "MLPClassifier (Adam)"]
    display_names = {
        "SGD logistic (log_loss)": "SGD 逻辑回归（Log Loss）",
        "SGD linear SVM (hinge)": "SGD 线性 SVM（Hinge Loss）",
        "MLPClassifier (Adam)": "MLP（Adam）",
    }
    for row_index, name in enumerate(order):
        current = curve_df[curve_df["experiment"] == name]
        batch_rows = current[current["granularity"] == "batch"]
        epoch_rows = current[current["granularity"] == "epoch"]

        axes[row_index, 0].plot(
            batch_rows["x_epoch"],
            batch_rows["train_loss"],
            color="#9aa7b5",
            alpha=0.32,
            linewidth=0.7,
            label="训练批次损失（展示波动）",
        )
        axes[row_index, 0].plot(
            epoch_rows["epoch"], epoch_rows["train_loss"], marker=".", label="完整训练集损失"
        )
        axes[row_index, 0].plot(
            epoch_rows["epoch"], epoch_rows["validation_loss"], marker=".", label="验证集损失"
        )
        loss_label = "对数损失" if name != "SGD linear SVM (hinge)" else "Hinge 损失"
        axes[row_index, 0].set_title(f"{display_names[name]}：{loss_label}")
        axes[row_index, 0].set_xlabel("训练轮次（批次点以小数表示）")
        axes[row_index, 0].set_ylabel("损失")
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 0].legend(fontsize=8)

        axes[row_index, 1].plot(
            batch_rows["x_epoch"],
            batch_rows["train_accuracy"],
            color="#9aa7b5",
            alpha=0.32,
            linewidth=0.7,
            label="训练批次准确率（展示波动）",
        )
        axes[row_index, 1].plot(
            epoch_rows["epoch"], epoch_rows["train_accuracy"], marker=".", label="完整训练集准确率"
        )
        axes[row_index, 1].plot(
            epoch_rows["epoch"], epoch_rows["validation_accuracy"], marker=".", label="验证集准确率"
        )
        axes[row_index, 1].set_title(f"{display_names[name]}：准确率")
        axes[row_index, 1].set_xlabel("训练轮次（批次点以小数表示）")
        axes[row_index, 1].set_ylabel("准确率")
        axes[row_index, 1].set_ylim(0, 1.02)
        axes[row_index, 1].grid(alpha=0.25)
        axes[row_index, 1].legend(fontsize=8)

    fig.suptitle(
        "逐轮训练曲线：浅色点为批次指标；SGD 曲线是辅助模型，不代表最终 SVC / LogisticRegression 轨迹",
        fontsize=13,
    )
    fig.savefig(out_dir / "epoch_curves.png", dpi=180)
    plt.close(fig)


def run_cleaning_ablation(searches, X_train, y_train, X_val, y_val, out_dir: Path):
    conditions = [
        ("raw", "原始文本"),
        ("keep_subject", "删除非 Subject 邮件头"),
        ("body_only", "删除全部邮件头（含 Subject）"),
        ("without_quotes", "删除引用回复行"),
        ("without_signatures", "删除签名块"),
    ]
    rows = []
    for condition, label in conditions:
        train_docs = clean_documents(X_train, condition)
        val_docs = clean_documents(X_val, condition)
        print(f"\n[清洗消融] 开始：{label}", flush=True)

        for model_name in CORE_MODEL_NAMES:
            search = searches[model_name]
            model = clone(search.best_estimator_)
            model.fit(train_docs, y_train)
            train_metrics = evaluate(model, train_docs, y_train)
            val_metrics = evaluate(model, val_docs, y_val)
            rows.append(
                {
                    "condition": condition,
                    "condition_label": label,
                    "model": model_name,
                    "train_accuracy": train_metrics["accuracy"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "validation_accuracy": val_metrics["accuracy"],
                    "validation_macro_f1": val_metrics["macro_f1"],
                    "validation_loss_name": val_metrics["loss_name"],
                    "validation_loss": val_metrics["loss"],
                    "n_features": len(model.named_steps["tfidf"].vocabulary_),
                }
            )
            print(
                f"  {model_name}："
                f"验证集 Accuracy={val_metrics['accuracy']:.4f}，"
                f"Macro-F1={val_metrics['macro_f1']:.4f}",
                flush=True,
            )

        print(f"[清洗消融] 完成：{label}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "cleaning_ablation.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    pivot = result.pivot(index="condition_label", columns="model", values="validation_accuracy")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("文本清洗消融：验证集准确率（固定原始文本搜索所得参数）")
    ax.set_xlabel("文本处理条件")
    ax.set_ylabel("验证集准确率")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(out_dir / "cleaning_ablation.png", dpi=180)
    plt.close(fig)
    return result


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def main() -> None:
    warnings.simplefilter("once", ConvergenceWarning)
    configure_chinese_plots()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[阶段 1/5] 读取并检查有标签数据", flush=True)
    texts, labels = load_labeled_data()
    print_labeled_data_preview(texts, labels)
    audit_target_headers(texts)
    if len(texts) != len(labels):
        raise ValueError("Training text and target counts do not match")
    if any(not text.strip() for text in texts):
        raise ValueError("Empty training text found; resolve it before running the experiment")

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    X_train = [texts[index] for index in train_idx]
    y_train = labels[train_idx]
    X_val = [texts[index] for index in val_idx]
    y_val = labels[val_idx]
    cv = make_cv()

    print(
        f"[数据划分] 模型训练 {len(train_idx)} 条，"
        f"留出验证 {len(val_idx)} 条；随机种子 {RANDOM_STATE}。",
        flush=True,
    )

    dump_json(
        OUTPUT_DIR / "experiment_config.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "random_state": RANDOM_STATE,
            "train_validation_split": "stratified 80:20",
            "cv": "StratifiedKFold(n_splits=3, shuffle=True, random_state=42)",
            "max_features": MAX_FEATURES,
            "min_df": MIN_DF,
            "sublinear_tf": SUBLINEAR_TF,
            "base_analyzer": "word",
            "base_ngram_range": BASE_NGRAM_RANGE,
            "mlp_grid": {
                "hidden_layer_sizes": [[50], [100]],
                "alpha": [0.0001, 0.00001],
            },
            "sgd_comparison_models": {
                name: {"loss": loss, "alpha_grid": [1e-5, 1e-4, 1e-3, 1e-2]}
                for name, loss in SGD_MODEL_LOSSES.items()
            },
            "curve_epochs": CURVE_EPOCHS,
            "curve_batch_size": 200,
            "feature_representation_ablation": "omitted by user decision",
            "train_rows": len(train_idx),
            "validation_rows": len(val_idx),
            "label_counts_all": dict(Counter(map(str, labels))),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    )
    dump_json(
        OUTPUT_DIR / "split_indices.json",
        {"train_indices": train_idx.tolist(), "validation_indices": val_idx.tolist()},
    )

    print("\n[阶段 2/5] 六模型三折交叉验证和留出验证", flush=True)
    searches, comparison = save_main_model_comparison(
        X_train, y_train, X_val, y_val, cv, OUTPUT_DIR
    )
    save_main_comparison_plot(comparison, OUTPUT_DIR)
    save_grid_search_plots(searches, OUTPUT_DIR)

    print("\n[阶段 3/5] 绘制六模型验证集混淆矩阵", flush=True)
    save_confusion_matrices(searches, X_val, y_val, OUTPUT_DIR)

    print("\n[阶段 4/5] 记录 SGD 与 MLP 逐轮曲线", flush=True)
    save_epoch_curves(searches, X_train, y_train, X_val, y_val, OUTPUT_DIR)

    print("\n[阶段 5/5] 比较文本清洗条件", flush=True)
    cleaning = run_cleaning_ablation(
        searches, X_train, y_train, X_val, y_val, OUTPUT_DIR
    )

    print("\n实验运行完成。", flush=True)
    print(f"输出目录：{OUTPUT_DIR}", flush=True)
    print(f"清洗消融结果行数：{len(cleaning)}", flush=True)

if __name__ == "__main__":
    main()