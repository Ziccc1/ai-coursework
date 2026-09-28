from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.base import clone

from experiment import (
    OUTPUT_DIR,
    classifier_of,
    clean_documents,
    load_labeled_data,
    load_unlabeled_test,
)

#根据模型比较与清洗消融结果确定的最终方案：逻辑回归+keep subject
FINAL_MODEL = "LogisticRegression"
FINAL_CLEANING = "keep_subject"

CLEANING_LABELS = {
    "raw": "原始文本",
    "keep_subject": "删除非 Subject 邮件头",
    "body_only": "删除全部邮件头（含 Subject）",
    "without_quotes": "删除非 Subject 邮件头并删除引用回复行",
    "without_signatures": "删除非 Subject 邮件头并删除签名块",
}


def validation_reference(
    model_key: str,
    cleaning_condition: str,
) -> dict[str, float | str] | None:
    """读取最终方案在留出验证集上的实验记录。"""
    if cleaning_condition == "raw":
        path = OUTPUT_DIR / "model_comparison.csv"
        results = pd.read_csv(path)
        rows = results[results["model"] == model_key]
    else:
        path = OUTPUT_DIR / "cleaning_ablation.csv"
        results = pd.read_csv(path)
        rows = results[
            (results["model"] == model_key)
            & (results["condition"] == cleaning_condition)
        ]

    if rows.empty:
        return None

    row = rows.iloc[0]
    return {
        "validation_accuracy": float(row["validation_accuracy"]),
        "validation_macro_f1": float(row["validation_macro_f1"]),
        "validation_loss_name": str(row["validation_loss_name"]),
        "validation_loss": float(row["validation_loss"]),
    }


def main() -> None:
    candidate_path = (
        OUTPUT_DIR / "models" / f"{FINAL_MODEL}_best_on_outer_train.joblib"
    )
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"请先运行 experiment.py；缺少候选模型：{candidate_path}"
        )

    texts, labels = load_labeled_data()
    test_texts = load_unlabeled_test()

    print("--- 无标签测试数据加载成功 ---", flush=True)
    print(f"无标签测试集样本数量: {len(test_texts)}", flush=True)
    print("-" * 20, flush=True)
    if test_texts:
        print("第一个需要预测的测试样本内容:", flush=True)
        print(test_texts[0], flush=True)
    print("\n" + "=" * 50, flush=True)

    print(
        f"[最终预测] 模型：{FINAL_MODEL}；"
        f"清洗条件：{CLEANING_LABELS[FINAL_CLEANING]}。",
        flush=True,
    )

    #对有标签数据和无标签测试数据应用相同的清洗规则
    train_texts = clean_documents(texts, FINAL_CLEANING)
    cleaned_test_texts = clean_documents(test_texts, FINAL_CLEANING)

    #读取实验选出的参数配置，用全部有标签数据重新拟合
    reviewed_candidate = joblib.load(candidate_path)
    final_model = clone(reviewed_candidate)
    final_model.fit(train_texts, labels)
    predictions = final_model.predict(cleaned_test_texts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    #输出单列、无表头的预测CSV
    prediction_path = OUTPUT_DIR / "predictions.csv"
    pd.DataFrame(predictions).to_csv(
        prediction_path,
        index=False,
        header=False,
        encoding="utf-8",
    )

    final_model_path = OUTPUT_DIR / "selected_final_model.joblib"
    joblib.dump(final_model, final_model_path)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": FINAL_MODEL,
        "cleaning_condition": FINAL_CLEANING,
        "cleaning_condition_label": CLEANING_LABELS[FINAL_CLEANING],
        "parameters": str(classifier_of(final_model).get_params()),
        "labeled_rows_used_for_refit": len(texts),
        "unlabeled_test_rows_predicted": len(test_texts),
        "validation_reference": validation_reference(
            FINAL_MODEL,
            FINAL_CLEANING,
        ),
        "prediction_file": str(prediction_path),
        "model_file": str(final_model_path),
    }
    (OUTPUT_DIR / "submission_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[最终预测] 已保存 {len(predictions)} 条预测：{prediction_path}",
        flush=True,
    )
    print(f"[最终预测] 最终模型已保存：{final_model_path}", flush=True)


if __name__ == "__main__":
    main()