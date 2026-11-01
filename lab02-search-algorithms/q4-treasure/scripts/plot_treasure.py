# 普通Matplotlib从Q4实际JSONL绘图；仅输出PNG，不使用绘图技能。
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from scripts.analyze_treasure import main as analyze, read_rows
from scripts.plot_results import COLORS, interval, setup

ROOT = Path(__file__).resolve().parents[1]
QOUT = ROOT/'outputs/q4'
FIG = ROOT/'outputs/figures'
MANIFEST = []
ALG = {'MM': 'Minimax', 'AB': 'Alpha-beta', 'AB+Order': 'AB+排序',
       'AB+TT': 'AB+缓存', 'AB+Order+TT': 'AB+排序+缓存'}


def save(fig, name, sources, claim):
    fig.savefig(FIG/f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    MANIFEST.append(dict(id=name, path=f'outputs/figures/{name}.png', source=sources,
                         claim=claim, script='scripts/plot_treasure.py'))


def teacher_figure():
    teacher = json.loads((QOUT/'diagnostics/teacher.json').read_text(encoding='utf-8'))
    lines = json.loads((QOUT/'diagnostics/complete_continuations.json').read_text(encoding='utf-8'))['teacher']
    fig, axes = plt.subplots(2, 1, figsize=(4.3, 4.7), layout='constrained')
    x = np.arange(4)
    for i, depth in enumerate([2, 4]):
        vals = list(teacher['by_depth'][str(depth)]['action_values'].values())
        bars = axes[0].bar(x+(i-.5)*.34, vals, width=.34, color=COLORS[i], label=f'H={depth}')
        for bar, v in zip(bars, vals):
            axes[0].annotate(str(v), (bar.get_x()+bar.get_width()/2, v),
                             xytext=(0, 3 if v >= 0 else -3), textcoords='offset points',
                             ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
    axes[0].axhline(0, color='#888888', lw=.7)
    axes[0].set(xticks=x, xticklabels=['T1', 'T2', 'T3', 'T4'], ylabel='该首步的回传值 U',
                title='a  同一首步在不同深度下的评价', ylim=(-8.5, 4.7))
    axes[0].legend(frameon=False, ncol=2)
    for key, label, color in [('optimal_line', '首步 T3，随后双方最优', COLORS[1]),
                               ('fixed_shallow_first_line', '首步 T2，随后双方最优', COLORS[0])]:
        line = lines[key]
        axes[1].plot(range(5), [0]+[r['utility'] for r in line], marker='o', color=color, label=label)
    axes[1].set(xticks=range(5), xlabel='已执行行动数', ylabel='累计收益差 U',
                title='b  在完整四次行动下比较首步决策')
    axes[1].legend(frameon=False, fontsize=7, loc='upper left')
    save(fig, 'q4_teacher', 'Q4-01; full-window diagnostics', '深度改变原题首步；浅层首步的完整损失为2，而非不同深度估值相减')


def scenario_figure(summary=None):
    paths = json.loads((QOUT/'diagnostics/complete_continuations.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.8))
    fig.subplots_adjust(left=.035, right=.965, bottom=.08, top=.94, wspace=.12, hspace=.19)
    names = {'C1': '短期诱惑', 'C2': '通道枢纽',
             'C3': '双方争抢', 'C4': '高代价绕行'}
    terrain_colors = ['#26333D', '#F4F4EE', '#D8AC6E']
    for ax, name in zip(axes.flat, ['C1', 'C2', 'C3', 'C4']):
        data = paths[name]
        case = data['case']
        assert set(v for row in case['grid'] for v in row) <= {None, 1, 9}
        cells = np.array([[0 if v is None else 1 if v == 1 else 2
                           for v in row] for row in case['grid']])
        ax.imshow(cells, cmap=ListedColormap(terrain_colors),
                  vmin=0, vmax=2, interpolation='nearest')
        for label, (r, c) in case['points'].items():
            if label in ('A', 'B'):
                ax.text(c, r, label, ha='center', va='center',
                        color='#20252A', fontsize=9, zorder=4)
            else:
                ax.text(c, r, f'{label}\n({case["values"][label]})', ha='center',
                        va='center', fontsize=8.5, linespacing=.72,
                        color='#20252A', zorder=4)
        ax.set_title(f'{name} {names[name]}', fontsize=10.5, pad=6)
        ax.set(xticks=[], yticks=[])
        ax.set_xticks(np.arange(-.5, 9, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 9, 1), minor=True)
        ax.grid(which='minor', color='#C9CCC8', linewidth=.45)
        ax.tick_params(which='minor', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.legend(handles=[Patch(facecolor=terrain_colors[1], label='普通格（代价1）'),
                        Patch(facecolor=terrain_colors[0], label='障碍'),
                        Patch(facecolor=terrain_colors[2], label='高代价格（代价9）')],
               loc='lower center', bbox_to_anchor=(.52, .005),
               ncol=3, frameon=False, fontsize=9, handlelength=1.2)
    save(fig, 'q4_cases', 'outputs/q4/diagnostics/complete_continuations.json',
         '四类场景的玩家位置、宝藏价值与地形布局；宝藏编号下方数字表示价值')


def ablation_figure(frame):
    data = frame[(frame.experiment_id == 'Q4-03') & (frame.analysis_role == 'ablation')]
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 4.8), layout='constrained')
    for i, algorithm in enumerate(ALG):
        sub = data[(data.algorithm == algorithm) & (data.status == 'ok')]
        for ax, metric, factor in [(axes[0], 'calls', 1), (axes[1], 'elapsed_ns', 1e-6)]:
            interval(ax, [4, 6, 8], [sub[sub.depth == d][metric].to_numpy()*factor for d in [4, 6, 8]],
                     ALG[algorithm], COLORS[i], ['o', 's', '^', 'D', 'v'][i])
    axes[0].set(ylabel='递归访问次数（对数轴）', yscale='log', title='a  排序与缓存的消融对照')
    axes[1].set(xlabel='搜索深度 H', ylabel='搜索时间 / ms（对数轴）', yscale='log',
                title='b  计入排序、缓存与规范首步重搜成本')
    for ax in axes:
        ax.set_xticks([4, 6, 8])
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    save(fig, 'q4_ablation', 'Q4-03; N=20 per condition', '节点减少与实际耗时共同评价，五版本保持同一有限深度目标')


def ordering_figure(frame):
    data = frame[(frame.experiment_id == 'Q4-03') & (frame.depth == 6) & frame.pruning & ~frame.cache]
    baseline = data[data.ordering == 'fixed'].set_index('seed')
    selection = json.loads((QOUT/'ordering_selection.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(2, 1, figsize=(4.3, 4.3), layout='constrained')
    for ax, metric, title in [(axes[0], 'calls', 'a  相对固定顺序的访问次数'),
                               (axes[1], 'elapsed_ns', 'b  相对固定顺序的搜索时间')]:
        arrays = []
        for order in ['immediate', 'competition']:
            if order == 'immediate':
                sub = data[(data.ordering == order) | ((data.ordering == 'competition') & (data.eta == 0))]
            else:
                sub = data[(data.ordering == order) & (data.eta > 0)]
            sub = sub.set_index('seed').loc[baseline.index]
            arrays.append((sub[metric]/baseline[metric]).to_numpy())
        box = ax.boxplot(arrays, tick_labels=['即时净收益', f'竞争感知（η={selection["positive_eta"]:g}）'], patch_artist=True,
                         medianprops={'color': '#183442', 'linewidth': 1.4})
        for patch, color in zip(box['boxes'], COLORS[:2]):
            patch.set(facecolor=color, alpha=.35)
        ax.axhline(1, color='#777777', lw=1, ls=':')
        ax.set(ylabel='同实例配对比值', title=title)
    save(fig, 'q4_ordering', 'Q4-03; test seeds 10..29; parameter frozen on development seeds',
         '排序收益在独立测试集配对比较，不免费使用预知子树值')


def sensitivity_figure():
    rows = json.loads((QOUT/'diagnostics/value_sweeps.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(2, 2, figsize=(4.7, 4.5), layout='constrained')
    for i, case in enumerate(['teacher', 'C3']):
        for j, depth in enumerate([2, 4]):
            ax = axes[i, j]
            data = sorted([r for r in rows if r['case'] == case and r['depth'] == depth], key=lambda r: r['value'])
            for k, target in enumerate(['T1', 'T2', 'T3', 'T4']):
                ax.plot([r['value'] for r in data], [r['action_values'][target] for r in data],
                        color=COLORS[k], lw=1.2, label=target)
            xs, ys = [], []
            for row in data:
                xs.append(row['value'])
                ys.append(row['best_value'])
            ax.scatter(xs, ys, s=8, facecolors='none', edgecolors='#20262C', linewidths=.5, zorder=4)
            target = 'T3' if case == 'teacher' else 'T2'
            ax.set(title=f'{"原题" if case == "teacher" else "C3"}，H={depth}', xlabel=f'{target} 的价值', ylabel='首步回传值 U')
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(frameon=False, fontsize=6.5, ncol=2)
    save(fig, 'q4_sensitivity', 'Q4-04; integer scans; ties retained', '最优首步随宝藏价值改变，空心点表示最优值包络')


def scaling_figures(frame):
    data = frame[(frame.experiment_id == 'Q4-05') & (frame.status == 'ok')]
    fig, axes = plt.subplots(3, 1, figsize=(4.4, 6.4), layout='constrained')
    sub = data[(data.axis == 'map_size') & (data.algorithm == 'AB+Order+TT')]
    for metric, color, label in [('preprocess_ns', COLORS[0], 'A*距离预计算'), ('elapsed_ns', COLORS[1], '优化博弈搜索')]:
        interval(axes[0], [7, 15, 30, 50], [sub[sub.axis_value == x][metric].to_numpy()*1e-6 for x in [7, 15, 30, 50]], label, color)
    axes[0].set(xlabel='地图边长', ylabel='时间 / ms（对数轴）', yscale='log', title='a  分解寻路与博弈开销')
    axes[0].legend(frameon=False, fontsize=7)
    for ax, axis, xs, title, xlabel in [(axes[1], 'treasure_count', [4, 6, 8, 10, 12], 'b  固定 H=4，增加宝藏数量', '宝藏数量'),
                                        (axes[2], 'depth', [2, 4, 6, 8], 'c  固定10个宝藏，增加搜索深度', '搜索深度 H')]:
        for i, algorithm in enumerate(['MM', 'AB', 'AB+Order+TT']):
            sub = data[(data.axis == axis) & (data.algorithm == algorithm)]
            interval(ax, xs, [sub[sub.axis_value == x].elapsed_ns.to_numpy()*1e-6 for x in xs], ALG[algorithm], COLORS[i], ['o', 's', '^'][i])
        ax.set(xlabel=xlabel, ylabel='搜索时间 / ms（对数轴）', yscale='log', title=title)
        ax.set_xticks(xs)
    axes[1].legend(frameon=False, fontsize=7)
    save(fig, 'q4_scaling', 'Q4-05; N=20 per condition; variables controlled separately',
         '网格大小影响距离预计算，宝藏分支与深度影响博弈搜索，两部分开销分别测量')

    fig, axes = plt.subplots(2, 1, figsize=(4.3, 4.5), layout='constrained')
    for i, algorithm in enumerate(['MM', 'AB', 'AB+Order+TT']):
        sub = data[(data.axis == 'full_search_reference') & (data.algorithm == algorithm)]
        for ax, metric, scale in [(axes[0], 'calls', 1), (axes[1], 'elapsed_ns', 1e-6)]:
            interval(ax, [4, 6, 8], [sub[sub.axis_value == x][metric].to_numpy()*scale for x in [4, 6, 8]], ALG[algorithm], COLORS[i])
    axes[0].set(ylabel='递归访问次数（对数轴）', yscale='log', title='a  完整搜索 H=宝藏数量')
    axes[1].set(xlabel='宝藏数量', ylabel='搜索时间 / ms（对数轴）', yscale='log', title='b  与固定深度规模实验分开解释')
    for ax in axes:
        ax.set_xticks([4, 6, 8])
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, 'q4_full_search', 'Q4-05; full_search_reference; N=20', '完整搜索规模增长不与固定H的分支实验混为同一趋势')


def main():
    analyze()
    setup()
    frame = read_rows()
    summary = json.loads((QOUT/'analysis.json').read_text(encoding='utf-8'))
    teacher_figure()
    scenario_figure(summary)
    ablation_figure(frame)
    ordering_figure(frame)
    sensitivity_figure()
    scaling_figures(frame)
    (QOUT/'figure_manifest.json').write_text(json.dumps(MANIFEST, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Created {len(MANIFEST)} Q4 PNG figures')


if __name__ == '__main__':
    main()
