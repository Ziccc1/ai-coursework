"""单独运行 TF-IDF 设置消融"""
from __future__ import annotations
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from experiment import (
    OUTPUT_DIR,
    RANDOM_STATE,
    clean_documents,
    configure_chinese_plots,
    load_labeled_data,
    make_pipeline,
)

# 与已经选定的最终方案保持一致，每组只改变一个 TF-IDF 参数。
SETTINGS = [
    ("baseline", "基准：词 1–2 元", {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}),
    ("unigram", "仅词 1 元", {"ngram_range": (1, 1), "min_df": 2, "sublinear_tf": True}),
    ("min_df_1", "min_df=1", {"ngram_range": (1, 2), "min_df": 1, "sublinear_tf": True}),
    ("linear_tf", "不使用次线性词频", {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": False}),
]

def main() -> None:
    configure_chinese_plots()
    split_path = OUTPUT_DIR / "split_indices.json"
    texts, labels = load_labeled_data()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_idx = np.asarray(split["train_indices"], dtype=int)
    val_idx = np.asarray(split["validation_indices"], dtype=int)
    X_train = clean_documents([texts[i] for i in train_idx], "keep_subject")
    X_val = clean_documents([texts[i] for i in val_idx], "keep_subject")
    y_train, y_val = labels[train_idx], labels[val_idx]
    rows = []

    for key, label, params in SETTINGS:
        model = make_pipeline(
            LogisticRegression(C=100.0, max_iter=2000, random_state=RANDOM_STATE),
            ngram_range=params["ngram_range"],
        )
        #向量器在Pipeline.fit内只从训练部分学习词表和IDF
        model.set_params(
            tfidf__min_df=params["min_df"],
            tfidf__sublinear_tf=params["sublinear_tf"],
        )
        print(f"[TF-IDF 消融] 开始：{label}", flush=True)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        row = {
            "setting": key,
            "setting_label": label,
            "model": "LogisticRegression",
            "cleaning": "keep_subject",
            "C": 100.0,
            "max_features": model.named_steps["tfidf"].max_features,
            "ngram_range": str(params["ngram_range"]),
            "min_df": params["min_df"],
            "sublinear_tf": params["sublinear_tf"],
            "n_features": len(model.named_steps["tfidf"].vocabulary_),
            "train_accuracy": accuracy_score(y_train, train_pred),
            "validation_accuracy": accuracy_score(y_val, val_pred),
            "validation_macro_f1": f1_score(y_val, val_pred, average="macro"),
        }
        rows.append(row)
        print(
            f"[TF-IDF 消融] 完成：{label}；"
            f"验证集 Accuracy={row['validation_accuracy']:.4f}，"
            f"Macro-F1={row['validation_macro_f1']:.4f}，"
            f"实际特征数={row['n_features']}",
            flush=True,
        )

    out_dir = OUTPUT_DIR / "tfidf_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "tfidf_ablation.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    y = np.arange(len(result))
    ax.barh(y - 0.18, result["validation_accuracy"], height=0.34, label="准确率", color="#315b89")
    ax.barh(y + 0.18, result["validation_macro_f1"], height=0.34, label="Macro-F1", color="#c8794b")
    ax.set_yticks(y, result["setting_label"])
    ax.invert_yaxis()
    ax.set_xlim(0.85, 0.96)
    ax.set_xlabel("验证集分数")
    ax.set_title("TF-IDF 单因素消融：逻辑回归 + 保留 Subject")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend()
    fig.savefig(out_dir / "tfidf_ablation.png", dpi=180)
    plt.close(fig)
    print(f"结果已保存至：{out_dir}", flush=True)

if __name__ == "__main__":
    main()
