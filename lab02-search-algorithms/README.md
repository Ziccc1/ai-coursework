# 实验二：A*、Minimax 与 Alpha-Beta 搜索

本项目包含实验二的四道题目。四道题分别放在独立文件夹中，每题都能单独复制运行，包含基础任务、拓展实验、给定样例、正确性测试、正式实验结果及 PNG 图表。

## 第一次运行：以题目一为例

在终端中进入项目目录，再进入 `q1-maze` 文件夹。下面的命令均在该题目文件夹内执行。

### 1. 运行给定样例，查看答案

```powershell
python main.py
```

程序会在终端输出给定地图的求解结果。题目一分别运行零启发、欧氏距离和曼哈顿距离；`path` 是路径坐标，`cost` 是移动步数，`expanded` 是扩展节点数。给定样例的最短步数为7。

其他三题的操作相同：进入对应文件夹，执行 `python main.py` 即可。此命令只运行本题给定样例，不会启动全部拓展实验。

### 2. 运行对照实验，生成结果数据

先运行一组快速实验，熟悉结果位置：

```powershell
python main.py experiments --ids Q1-01 --timing single
```

`experiments` 表示运行实验；`--ids Q1-01` 选择题目一的第一组实验；`--timing single` 每条配置只求解一次，适合检查结果。运行后查看本题的 `outputs_reproduced/raw/Q1-01.jsonl`，每行对应一条配置的结果。

需要重新完成本题所有实验并测量耗时时，运行：

```powershell
python main.py experiments --ids all --timing full
```

`all` 表示本题全部实验组，`full` 使用重复计时流程，因此会比前面的快速实验耗时更长。修改代码或计时模式后，请将旧的 `outputs_reproduced/` 另行保存，再开始新一轮运行。

### 3. 根据随附结果生成PNG图片

首次绘图前安装依赖：

```powershell
python -m pip install -r requirements.txt
python main.py plots
```

图片保存在本题 `outputs/figures/` 中。当前 `plots` 命令读取随项目提供的正式数据 `outputs/`，不自动读取刚刚重跑生成的 `outputs_reproduced/`。

### 文件之间如何配合

`main.py` 负责接收命令；样例求解通过 `algorithm.py` 调用 `searchlab/` 中的算法；对照实验由实验模块生成输入、运行不同配置并保存数据；绘图命令调用 `scripts/` 中的程序读取结果并生成图片。

| 你想做什么 | 在题目文件夹内运行 | 查看哪里 |
|---|---|---|
| 看给定样例的答案 | `python main.py` | 终端输出 |
| 求解自己的输入 | `python main.py solve --input 输入文件.json` | 终端输出；输入格式参考 `inputs/` |
| 重跑本题全部实验 | `python main.py experiments --ids all --timing full` | `outputs_reproduced/` |
| 生成随附结果的图片 | `python main.py plots` | `outputs/figures/` |
| 检查随附结果 | `python main.py check` | 终端输出的核验结果 |
| 查看全部参数 | `python main.py --help` | 终端输出 |

## 目录

```text
lab02-search-algorithms/
├── README.md                   总说明与运行方法
├── requirements.txt            汇总、绘图依赖
├── verify.py                   全部实验的答案与统计一致性核验
├── verification/               核验报告和文件校验清单
├── q1-maze/                    题目一：迷宫寻路
├── q2-weighted/                题目二：加权地形
├── q3-alpha-beta/              题目三：Minimax、Alpha-Beta 与缓存
└── q4-treasure/                题目四：A*距离矩阵与宝藏竞争博弈
```

每题均包含运行入口、算法实现、输入样例、正确性测试、实验脚本和结果。各题可独立运行，也可在本目录下通过相对路径调用。

## 分题说明与实验内容

点击下表中的链接，可查看各题的运行命令、参数、输入格式及结果目录。

| 题目 | 详细说明 | 实验组 |
|---|---|---|
| 题目一：迷宫寻路 | [查看题目一 README](q1-maze/README.md) | Q1-01—Q1-04 |
| 题目二：加权地形 | [查看题目二 README](q2-weighted/README.md) | Q2-01—Q2-04 |
| 题目三：博弈树求值与剪枝 | [查看题目三 README](q3-alpha-beta/README.md) | Q3-01—Q3-04 |
| 题目四：竞争环境下的宝藏分配 | [查看题目四 README](q4-treasure/README.md) | Q4-01—Q4-05 |

### 题目一：迷宫寻路

- **基础任务**：在四方向、单位移动代价的地图中使用 A* 求最短路径，返回包含起终点的坐标序列；不可达时返回空列表。比较零启发、欧氏距离和曼哈顿距离，并记录扩展节点数与耗时。
- **拓展实验**：比较较大 `g` 优先、较小 `g` 优先等平局规则；设置开放地图、蛇形走廊、梳状死路和强制绕行等结构，并改变地图规模与障碍概率，分析搜索效率受哪些条件影响。

### 题目二：加权地形

- **基础任务**：按进入格子的地形权重累计路径代价，起点不计代价、进入终点代价为0，使用 A* 返回最小累计代价；不可达时返回 `-1`。根据终点规则修正曼哈顿下界，并验证启发函数的可采纳性与一致性。
- **拓展实验**：使用高代价地形比较最少步数与最小代价路线；比较零启发、修正曼哈顿下界和放松地图下界，考察地形分布、零代价格，以及固定终点多次查询时的预处理复用收益。

### 题目三：博弈树求值与剪枝

- **基础任务**：基于给定节点结构实现 Minimax 和 Alpha-Beta，完成 MAX、MIN 节点的数值回传，核对根节点得分，并比较子节点访问顺序对剪枝和搜索开销的影响。
- **拓展实验**：改变树的深度和分支数，比较不同排序条件；构造同层共享节点的无环图，通过不同复用概率比较置换表开启前后的递归访问次数与耗时。

### 题目四：竞争环境下的宝藏分配

- **基础任务**：先使用 A* 计算 A、B 与四个宝藏之间的 `6×6` 距离矩阵，再建立包含双方位置、剩余宝藏、累计收益和当前玩家的博弈状态。使用 Minimax 和 Alpha-Beta 分别在 `H=2`、`H=4` 时求最优首步与回传值，并结合完整行动序列解释决策差异。
- **拓展实验**：比较行动排序、置换表及二者组合的效果；设计短期诱惑、通道枢纽、双方争抢和高代价绕行四类场景，并开展宝藏价值扫描及规模实验，分析竞争关系、搜索深度和状态复用对决策与效率的影响。

全部实验共17组、4199条配置。每条配置对应一个输入及其算法参数；各题既可运行给定样例，也可重新生成并求解拓展实验。

## 快速查看给定样例

Python 3.10+；核心算法、样例和核验只使用标准库。在本目录运行：

```powershell
python q1-maze/main.py
python q2-weighted/main.py
python q3-alpha-beta/main.py
python q4-treasure/main.py
```

| 题目 | 给定样例的预期结果 |
|---|---|
| 一 | 最短路径7步，包含起终点共8个坐标；三种启发函数结果均为7步 |
| 二 | 最小路径代价7；进入终点E的代价为0 |
| 三 | Minimax与Alpha-Beta根值均为5；默认分别调用15、14次 |
| 四 | H=2选择T2、U=-1；H=4选择T3、U=3；两种算法一致 |

第四题严格采用 `Gain(p, Ti) = Vi - d(p, Ti)` 与 `U = Score_A - Score_B`。H表示单方行动次数；达到深度或宝藏取完时用当前U评价。题目四特殊格进入代价为1，与题目二的终点规则不同。

## 题目要求的函数接口

命令行会额外输出节点数、耗时等实验指标。若只需要题目规定的返回值，可在对应题目文件夹中从 `algorithm.py` 导入以下函数：

| 题目 | 函数接口 | 返回内容 |
|---|---|---|
| 一 | `astar_maze(start, goal, grid)` | 路径坐标列表；不可达返回 `[]` |
| 二 | `astar_weighted(start, goal, grid)` | 最小累计代价；不可达返回 `-1` |
| 三 | `minimax(root)`、`alpha_beta(root)` | 根节点得分；`TreeNode` 同样可从 `algorithm.py` 导入 |
| 四 | `solve(data, depth=2, algorithm='MM')` | 包含距离矩阵、`best_target`、`best_value` 及搜索统计的字典 |

第四题的 `depth` 可设为2或4，`algorithm` 可设为 `MM` 或 `AB`。输入数据结构参考各题 `inputs/` 中的给定样例。

## 核对分题后是否保持一致

```powershell
python verify.py
```

此命令从四个分题入口重跑全部17组、4199条实验配置，将每条结果与随附原始正式记录比较。核验包括输入哈希、路径、代价、博弈值、首步选择，以及扩展数、递归次数和剪枝等确定性统计。另执行各题正确性测试、检查给定样例和文件清单。具体核对项及实际结果见[核验说明与记录](verification/RESULTS.md)。

核验重跑采用 single 模式，每种配置进行一次功能求解，不重新测量报告中的多批次耗时。耗时、计时批大小等随运行环境变化，排除在一致性判断之外；这次核验结果不用于替换正式性能数据。完整核验需要一定时间，会显示各题进度。

## 单题求解、实验与绘图

进入某个题目目录，使用对应 README 中的命令，例如：

```powershell
cd q4-treasure
python main.py solve --depth 2 --algorithm MM
python main.py solve --depth 4 --algorithm AB --ordering competition --cache
python -m unittest discover -s tests -v
python main.py check
python main.py experiments --ids all --timing full
pip install -r requirements.txt
python main.py plots
```

正式结果保存在各题 `outputs/`，包含原始 JSONL、输入实例、CSV 汇总和 PNG。重跑输出保存在各题 `outputs_reproduced/`，已由 Git 忽略；`plots` 与 `check` 默认读取正式结果。只有 `--timing full` 重跑才遵循报告中的正式计时流程，具体毫秒值仍会受机器和负载影响。

绘图脚本使用普通 Python 和 Matplotlib，输出 PNG 图片。
