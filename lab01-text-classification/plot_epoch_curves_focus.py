from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CURVE_PATH = OUTPUT_DIR / "epoch_curves.csv"
FIGURE_PATH = OUTPUT_DIR / "epoch_curves_focus.png"

COLORS = {
    "train": "#3569A8",
    "validation": "#D97706",
    "batch": "#E7A1A1",
    "grid": "#D9DEE5",
}


def configure_chinese_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    available = [name for name in candidates if name in installed]
    plt.rcParams["font.sans-serif"] = available + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    configure_chinese_font()
    frame = pd.read_csv(CURVE_PATH)
    frame["epoch"] = pd.to_numeric(frame["epoch"])
    frame["x_epoch"] = pd.to_numeric(frame["x_epoch"])
    for column in ("train_accuracy", "validation_accuracy", "train_loss", "validation_loss"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # SGD settles early; the MLP's validation-loss minimum occurs at epoch 62,
    # so its focused window must extend far enough to retain that turning point.
    models = [
        ("MLPClassifier (Adam)", "MLP（Adam）", 75),
        ("SGD logistic (log_loss)", "SGD 逻辑回归（log loss）", 20),
        ("SGD linear SVM (hinge)", "SGD 线性 SVM（hinge loss）", 20),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.subplots_adjust(top=0.88, bottom=0.08, hspace=0.40, wspace=0.14)
    fig.suptitle("训练曲线：聚焦各模型有变化的轮次", fontsize=17, fontweight="bold", y=0.985)

    for row, (model_key, display_name, max_epoch) in enumerate(models):
        subset = frame[frame["experiment"] == model_key]
        epochs = subset[subset["granularity"] == "epoch"].sort_values("epoch")
        batches = subset[
            (subset["granularity"] == "batch") & (subset["x_epoch"] <= max_epoch)
        ].sort_values("x_epoch")
        epochs = epochs[epochs["epoch"] <= max_epoch]
        acc_ax, loss_ax = axes[row]

        for ax, batch_metric, train_metric, validation_metric in (
            (acc_ax, "train_accuracy", "train_accuracy", "validation_accuracy"),
            (loss_ax, "train_loss", "train_loss", "validation_loss"),
        ):
            ax.plot(
                batches["x_epoch"],
                batches[batch_metric],
                color=COLORS["batch"],
                linewidth=0.65,
                alpha=0.62,
                zorder=5,
            )
            ax.plot(
                epochs["epoch"],
                epochs[train_metric],
                color=COLORS["train"],
                linewidth=1.65,
                marker="o",
                markersize=2.8,
                zorder=3,
            )
            ax.plot(
                epochs["epoch"],
                epochs[validation_metric],
                color=COLORS["validation"],
                linewidth=1.65,
                marker="o",
                markersize=2.8,
                zorder=4,
            )
            ax.set_xlim(0, max_epoch + 1)
            ax.set_xlabel("训练轮次")
            ax.grid(color=COLORS["grid"], linewidth=0.7, alpha=0.75)
            ax.spines[["top", "right"]].set_visible(False)

        acc_ax.set_title(f"{display_name}：准确率（第 1–{max_epoch} 轮）", fontsize=12)
        acc_ax.set_ylabel("准确率")
        acc_ax.set_ylim(0.80, 1.005)
        acc_ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

        best_accuracy = subset[subset["granularity"] == "epoch"].loc[
            lambda values: values["validation_accuracy"].idxmax()
        ]
        if int(best_accuracy["epoch"]) <= max_epoch:
            acc_ax.scatter(
                [best_accuracy["epoch"]],
                [best_accuracy["validation_accuracy"]],
                marker="*",
                s=105,
                color=COLORS["validation"],
                edgecolor="white",
                linewidth=0.6,
                zorder=6,
            )
            acc_ax.text(
                0.98,
                0.06,
                f"验证准确率最高：第 {int(best_accuracy['epoch'])} 轮，"
                f"{best_accuracy['validation_accuracy']:.2%}",
                transform=acc_ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#374151",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
            )

        loss_label = "log loss" if "hinge" not in model_key else "hinge loss"
        loss_ax.set_title(f"{display_name}：{loss_label}（第 1–{max_epoch} 轮）", fontsize=12)
        loss_ax.set_ylabel("损失值")
        best_loss = subset[subset["granularity"] == "epoch"].loc[
            lambda values: values["validation_loss"].idxmin()
        ]
        if int(best_loss["epoch"]) <= max_epoch:
            loss_ax.scatter(
                [best_loss["epoch"]],
                [best_loss["validation_loss"]],
                marker="D",
                s=42,
                color=COLORS["validation"],
                edgecolor="white",
                linewidth=0.6,
                zorder=6,
            )
            loss_ax.text(
                0.98,
                0.06,
                f"验证损失最低：第 {int(best_loss['epoch'])} 轮，"
                f"{best_loss['validation_loss']:.4f}",
                transform=loss_ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#374151",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
            )

    legend_items = [
        Line2D([0], [0], color=COLORS["train"], marker="o", lw=2, label="训练集：每轮结束"),
        Line2D([0], [0], color=COLORS["validation"], marker="o", lw=2, label="验证集：每轮结束"),
        Line2D([0], [0], color=COLORS["batch"], lw=1.2, label="浅色：小批次训练指标（展示抖动）"),
    ]
    fig.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 0.944), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.018,
        "展示区间按数据确定：两个 SGD 模型显示前 20 轮；MLP 显示前 75 轮，包含验证损失最低点（第 62 轮）。"
        "MLP 第 200 轮验证损失为 0.2533，高于最低值 0.2373。",
        ha="center",
        fontsize=9.5,
        color="#4B5563",
    )
    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
