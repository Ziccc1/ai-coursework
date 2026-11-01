# 题目三：博弈树剪枝

Minimax与Alpha-Beta；访问顺序、规模、重复状态置换表对照。

本题可独立运行。`main.py` 提供命令行入口，`algorithm.py` 提供求解接口，核心实现位于 `searchlab/tree.py`；实验生成、独立参考计算和绘图分别由相应模块完成。

## 运行

Python 3.10+；求解和测试仅使用标准库。绘图先安装依赖。

以下命令均在本题文件夹中执行。

### 运行给定样例

```powershell
python main.py
```

结果直接显示在终端；该命令只运行给定样例，不运行全部拓展实验。

### 求解指定输入

```powershell
python main.py solve --input inputs/q3_teacher.json
```

可参考 `inputs/` 中的格式准备自己的JSON文件，再通过 `--input` 指定。

### 检查正确性

```powershell
python -m unittest discover -s tests -v
python main.py check
```

第一条运行单元测试，第二条独立检查 `outputs/` 中随附的正式实验结果。

### 生成图片

```powershell
python -m pip install -r requirements.txt
python main.py plots
```

生成的PNG位于 `outputs/figures/`。绘图读取随附的 `outputs/`，不自动使用重跑目录的数据。

第三题可通过 --algorithm MM 或 AB、--ordering fixed/reverse/random 和 --cache 控制搜索。理想排序需要完整子树值，仅由正式实验运行器生成。

## 本题实验与结果

随附 1623 行正式结果，只包含 Q3 实验组。`outputs/raw/` 保存完整计时与指标，`outputs/instances/` 保存本题使用的输入，`outputs/results.csv` 是本题结果表，`outputs/figures/` 是本题 PNG。

```powershell
python main.py experiments --ids all --timing full
python main.py experiments --ids Q3-01 --timing full --resume
```

首次可以用 `python main.py experiments --ids Q3-01 --timing single` 快速检查一组实验；正式性能测量使用上面的 `full` 模式。

重跑结果写入本目录 `outputs_reproduced/`；`--resume` 只复用该重跑目录中的记录。原正式结果保存在 `outputs/`，`plots` 和 `check` 默认读取正式结果。`--timing single` 仅用于调试。重新计时会随运行环境变化，不能要求与原始耗时完全相同。

## 文件索引

- `main.py`：本题命令行。
- `algorithm.py`：本题求解接口。
- `searchlab/`：算法实现、实验生成器与独立参考。
- `inputs/`：本题给定样例。
- `tests/`：本题正确性测试。
- `scripts/`：普通 Python 绘图与结果核验。
- `outputs/`：正式数据、实例和 PNG。

共享子树按父子关系递归访问；开启缓存后复用已求解的子问题。

## 完整核验

在项目根目录运行 `python verify.py`，重新计算全部17组配置，并逐条比较答案和确定性的搜索统计。核验记录见根目录 `verification/RESULTS.md`。
