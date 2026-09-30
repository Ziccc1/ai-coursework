# 实验一：新闻文本分类

本项目的实验任务是根据新闻文本预测10个类别，实验比较朴素贝叶斯、线性 SVC、逻辑回归、单隐藏层MLP，以及两个使用随机梯度下降（SGD）的线性分类器；另外分析训练曲线、文本清洗和 TF-IDF 设置的影响。最终预测方案为**逻辑回归 + 删除非 Subject 邮件头**，沿用主实验的词 1–2 元 TF-IDF 设置。

## 1. 文件说明

| 文件或目录                   | 作用                                                                                                                                   |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `experiment.py`              | 读取有标签数据，划分训练集和验证集，完成参数搜索、六模型比较、逐轮曲线、混淆矩阵和清洗消融。**不读取无标签测试集，也不生成最终预测。** |
| `plot_epoch_curves_focus.py` | 读取 `outputs/epoch_curves.csv`，绘制聚焦训练前期变化的曲线；不重新训练模型。                                                          |
| `tfidf_ablation.py`          | 沿用主实验的训练/验证划分，单独比较四组 TF-IDF 设置；不重跑六模型主实验，也不生成测试集预测。                                          |
| `final.py`                   | 使用已确定的逻辑回归参数和清洗规则，在全部有标签数据上重新拟合，再对无标签测试集预测。                                                 |
| `train_data.csv`             | 有标签数据，包含 `text` 和 `target` 两列。                                                                                             |
| `test_data_unlabeled.csv`    | 待预测数据，包含 `text` 列。                                                                                                           |
| `TUNING.md`                  | 课程提供的参数说明参考材料。                                                                                                           |
| `outputs/`                   | 实验生成的 CSV、图片、候选模型和最终预测文件。                                                                                         |

## 2. 环境与安装

本次实验实际运行的环境为 Windows、Python 3.10.21，主要依赖版本如下：

| 包           | 已运行版本 |
| ------------ | ---------- |
| NumPy        | 2.2.6      |
| pandas       | 2.3.3      |
| SciPy        | 1.15.2     |
| scikit-learn | 1.7.2      |
| Matplotlib   | 3.10.9     |
| joblib       | 1.6.0      |

可以使用 Conda 创建独立环境，并安装与本次运行一致的版本：

```powershell
conda create -n contemporary-ai-lab1 python=3.10 -y
conda activate contemporary-ai-lab1
python -m pip install numpy==2.2.6 pandas==2.3.3 scipy==1.15.2 scikit-learn==1.7.2 matplotlib==3.10.9 joblib==1.6.0
```

绘图脚本会优先使用系统中可用的中文字体，若图片中文字显示为方框，请先安装一种中文字体，再重新运行绘图脚本。

## 3. 运行方法

在包含实验脚本及两个数据 CSV 的目录中运行主实验和最终预测：

```powershell
python experiment.py
python plot_epoch_curves_focus.py
python final.py
```

三个命令的作用依次是：

1. `experiment.py` 完成全部训练和验证实验，将过程数据、图表和候选模型写入 `outputs/`。六模型网格搜索，尤其是线性 SVC 和 MLP，可能需要较长时间。终端会按模型输出搜索结果，并按轮输出曲线实验的训练与验证指标。
2. `plot_epoch_curves_focus.py` 根据已有的 `epoch_curves.csv` 生成 `outputs/epoch_curves_focus.png`。必须在第一步完成后运行。
3. `final.py` 读取第一步保存的逻辑回归候选模型以取得参数设置，将已确定的清洗规则同时应用于有标签数据和无标签测试数据，用全部有标签数据重新拟合，生成 `outputs/predictions.csv`。必须在第一步完成后运行。

脚本以自身所在目录定位输入和输出文件，无须修改绝对路径。若只想复现图表，可不运行 `final.py`；若只想重新生成聚焦曲线图，不必重跑 `experiment.py`。

TF-IDF 消融是后来增加的独立补充实验。首次运行前需通过 `python experiment.py` 生成 `outputs/split_indices.json`；如果仓库中已有该文件，可以直接运行：

```powershell
python tfidf_ablation.py
```

该命令只重新拟合四组逻辑回归与 TF-IDF 流水线，结果写入 `outputs/tfidf_ablation/`。它不修改主实验结果或 `outputs/predictions.csv`，也不需要运行 `final.py`。

## 4. 实验设置

### 4.1 数据划分与防止信息泄露

- 有标签数据共 **7368** 条，采用分层随机 **80:20** 划分：训练部分 **5894** 条，留出验证集 **1474** 条。随机种子固定为 `42`，划分索引保存于 `outputs/split_indices.json`。
- 仅在训练部分进行三折 `StratifiedKFold` 交叉验证，以平均准确率选择参数；同时记录 Macro-F1。留出验证集用于比较模型和清洗条件，不参与模型拟合或交叉验证。
- `TfidfVectorizer` 与分类器放在同一个 scikit-learn `Pipeline` 中。交叉验证每一折只在该折训练数据上拟合词表和 IDF，验证折仅调用 `transform`；留出验证集也仅调用 `transform`。
- 无标签测试集不用于调参。确定最终模型和清洗方案后，`final.py` 才读取它并生成预测。
- 实验会检查邮件头中是否存在直接表示类别的字段，避免明显的标签泄露。此检查不等同于删除全部邮件头；不同邮件头处理方式由清洗消融单独比较。

### 4.2 文本表示与预处理

六模型主比较统一使用词级 TF-IDF，设置为 `ngram_range=(1, 2)`、`max_features=20000`、`min_df=2`、`sublinear_tf=True`。其余未显式设置的选项采用本环境中 `TfidfVectorizer` 的默认值。主比较使用原始文本，不额外执行停用词删除、stemming 或标点删除；脚本也**没有**按是否含中文自动切换字符 n-gram。主比较阶段统一文本表示，使模型之间的分数具有可比性；后续消融分别改变清洗条件或 TF-IDF 设置。

### 4.3 模型、搜索范围与指标

| 模型                             | 搜索参数                                                                |
| -------------------------------- | ----------------------------------------------------------------------- |
| MultinomialNB                    | `alpha ∈ {0.1, 1, 10}`                                                  |
| 线性 SVC                         | `C ∈ {0.1, 1, 10, 100}`                                                 |
| LogisticRegression               | `C ∈ {0.1, 1, 10, 100}`，`max_iter=2000`                                |
| 单隐藏层 MLP                     | 隐藏层宽度 `{50, 100}` × `alpha ∈ {1e-4, 1e-5}`，共四组；`max_iter=200` |
| SGDClassifier，`loss="log_loss"` | `alpha ∈ {1e-5, 1e-4, 1e-3, 1e-2}`                                      |
| SGDClassifier，`loss="hinge"`    | `alpha ∈ {1e-5, 1e-4, 1e-3, 1e-2}`                                      |

六个模型均报告训练集准确率、留出验证集准确率与 Macro-F1，并保存三折 CV 的均值及标准差。具有概率输出的模型记录 log loss；具有决策分数的模型记录多分类 hinge loss。**不同定义的损失值不直接横向比较大小**，主要用来观察同一模型在训练和验证过程中的变化。

### 4.4 逐轮训练曲线

对选出的 SGD 参数和 MLP 参数，额外进行 **200 轮**训练，每个小批次最多 **200** 条样本。小批次指标用于展示训练波动；每轮结束后，在完整训练集与验证集上计算准确率和损失。因此，浅色的小批次曲线与逐轮的完整数据集曲线含义不同。

这里的 SGD 曲线对应两个实际参与六模型比较的 `SGDClassifier`。**它们不是 `LogisticRegression` 或 `SVC` 内部优化过程的逐轮轨迹。** MLP 曲线也是按选出的参数单独训练所得，用来分析收敛与过拟合趋势。

### 4.5 文本清洗消融

在四个课程模型上比较以下五种文本条件：原始文本、删除非 Subject 邮件头、删除全部邮件头（包括 Subject）、在保留 Subject 和正文的基础上删除引用回复行、在保留 Subject 和正文的基础上删除签名块。每种条件均对同一训练集与验证集应用相同规则，固定**原始文本搜索得到的模型参数**，重新拟合 TF-IDF 与分类器。清洗消融不再对每种条件单独搜索参数，因此其结果回答的是“在原参数下改变清洗方式有什么影响”。

### 4.6 TF-IDF 设置消融

补充实验沿用 `outputs/split_indices.json` 中的训练/验证划分，固定逻辑回归 `C=100`、删除非 Subject 邮件头的清洗方式，以及 `max_features=20000`。以主实验的词 1–2 元、`min_df=2`、`sublinear_tf=True` 为基准，每组只改变一项：仅用词 1 元、将 `min_df` 改为 1、或将 `sublinear_tf` 改为 `False`。每组均在训练部分重新拟合 TF-IDF 词表和分类器，再计算同一留出验证集的 Accuracy 与 Macro-F1。此步骤不重新搜索模型参数。

## 5. 本次结果与最终方案

下表为 `outputs/model_comparison.csv` 中的主实验结果。CV 准确率来自训练部分的三折交叉验证；验证集指标来自独立的留出验证集。

| 模型          | 最优参数                      | CV 准确率 | 验证集准确率 | 验证集 Macro-F1 |
| ------------- | ----------------------------- | --------: | -----------: | --------------: |
| SGD hinge     | `alpha=1e-4`                  |    92.42% |       93.15% |          93.17% |
| 逻辑回归      | `C=100`                       |    91.97% |       92.88% |          92.92% |
| SGD log loss  | `alpha=1e-5`                  |    92.26% |       92.81% |          92.85% |
| MLP           | `hidden=(100,)`, `alpha=1e-5` |    92.13% |       92.61% |          92.66% |
| 线性 SVC      | `C=1`                         |    91.16% |       92.33% |          92.42% |
| MultinomialNB | `alpha=1`                     |    89.40% |       89.01% |          89.11% |

原始文本的六模型比较中，SGD hinge 的留出验证集准确率最高。清洗消融中，**逻辑回归 + 删除非 Subject 邮件头**达到验证集准确率 **93.55%**、Macro-F1 **93.62%**，因而选为最终预测方案。`final.py` 固定该方案，将全部 **7368** 条有标签样本用于最后一次拟合，再预测 **2457** 条无标签样本。以上验证集分数用于方案选择，不能作为无标签测试集的真实分数。

补充的 TF-IDF 消融中，仅使用词 1 元时，验证集 Accuracy 为 **93.83%**、Macro-F1 为 **93.85%**；基准设置分别为 **93.55%** 和 **93.62%**。准确率差异对应 1474 条验证样本中多分对 4 条。这是在已查看同一验证集结果之后进行的探索，且新设置未重新进行完整的参数搜索与模型比较。因此本次提交保留已确定的词 1–2 元最终预测方案，将词 1 元的结果作为后续改进线索。

## 6. 输出文件与复现检查

| 路径                                                                                     | 内容                                                                                         |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `outputs/experiment_config.json`                                                         | 划分、特征、参数网格和运行环境记录。                                                         |
| `outputs/split_indices.json`                                                             | 本次训练与留出验证的原始行索引。                                                             |
| `outputs/model_comparison.csv`、`.png`                                                   | 六模型指标表与验证集对比图。                                                                 |
| `outputs/cv_results/*_grid.csv`、`outputs/grid_search_comparison.png`                    | 每组参数的交叉验证结果与参数对比图。                                                         |
| `outputs/epoch_curves.csv`、`outputs/epoch_curves.png`、`outputs/epoch_curves_focus.png` | 小批次及逐轮指标、完整曲线图和聚焦曲线图。聚焦图目前包含三种模型各一张准确率图和一张损失图。 |
| `outputs/cleaning_ablation.csv`、`.png`                                                  | 四模型 × 五种文本条件的消融结果。                                                            |
| `outputs/tfidf_ablation/tfidf_ablation.csv`、`.png`                                       | 固定逻辑回归与清洗条件后的四组 TF-IDF 设置消融结果及对比图。                                  |
| `outputs/confusion_matrices/*_validation.csv`、`.png`                                    | 六模型在**原始文本主实验**上的验证集混淆矩阵；不是最终清洗方案的混淆矩阵。                   |
| `outputs/models/*_best_on_outer_train.joblib`                                            | 训练部分拟合的各模型候选文件，供结果检查及 `final.py` 读取参数设置。                         |
| `outputs/selected_final_model.joblib`                                                    | 按最终方案在全部有标签数据上重新拟合的模型。                                                 |
| `outputs/submission_metadata.json`                                                       | 最终模型、清洗条件、样本数及预测文件位置。                                                   |
| **`outputs/predictions.csv`**                                                            | **实际提交的预测结果：2457 行、单列、无表头。**                                              |

在相同数据及上述环境中重新运行后，数据划分、六模型汇总、逐轮曲线数据、清洗消融数据和六张混淆矩阵的 CSV 与上一次运行一致；`outputs/predictions.csv` 也与上一次最终预测一致。

`outputs/experiment_config.json` 记录的是主实验当时的配置；TF-IDF 消融为其后新增的独立实验，参数和结果以 `tfidf_ablation.py` 及 `outputs/tfidf_ablation/` 为准。
