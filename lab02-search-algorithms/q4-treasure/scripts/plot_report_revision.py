# 用已保存的真实实验数据绘制报告精简版；普通 Matplotlib，仅导出 PNG。
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/q4/diagnostics'
OUT = ROOT / 'outputs/figures'
BLUE, ORANGE = '#3976A4', '#D78A42'
TARGET_COLORS = ['#597EA6', '#D58A36', '#388771', '#AF6584']


def setup():
    plt.rcParams.update({
        'font.family': ['Microsoft YaHei', 'DejaVu Sans'],
        'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11.5,
        'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#6F7780', 'axes.linewidth': .7,
        'axes.unicode_minus': False, 'savefig.facecolor': 'white',
    })


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=360, bbox_inches='tight', pad_inches=.07)
    plt.close(fig)


def bars(ax, teacher):
    x = np.arange(4)
    for j, (depth, color) in enumerate([(2, BLUE), (4, ORANGE)]):
        values = [teacher['by_depth'][str(depth)]['action_values'][f'T{i}'] for i in range(1, 5)]
        containers = ax.bar(x + (j - .5) * .35, values, .31, color=color,
                            label=f'$H={depth}$', zorder=3)
        for bar, value in zip(containers, values):
            ax.annotate(str(value), (bar.get_x() + bar.get_width()/2, value),
                        xytext=(0, 4 if value >= 0 else -4), textcoords='offset points',
                        ha='center', va='bottom' if value >= 0 else 'top',
                        fontsize=10.5, color='#26313A')
    ax.axhline(0, color='#737E86', lw=.9, zorder=2)
    ax.set(xticks=x, xticklabels=['T1', 'T2', 'T3', 'T4'],
           xlabel='候选首步', ylabel='回传值 $U$', ylim=(-8.6, 4.7), xlim=(-.6, 3.6))
    ax.set_yticks([-8, -6, -4, -2, 0, 2, 4])
    ax.legend(frameon=False, ncol=2, loc='upper left', handlelength=1.3,
              handletextpad=.4, columnspacing=1.1, borderaxespad=.25, fontsize=10)
    ax.tick_params(length=3)


def matrix(ax, teacher):
    labels = teacher['labels']
    values = teacher['matrix']
    assert labels == ['A', 'B', 'T1', 'T2', 'T3', 'T4']
    ax.set_axis_off()
    table = ax.table(cellText=[[str(v) for v in row] for row in values],
                     rowLabels=labels, colLabels=labels, cellLoc='center',
                     rowLoc='center', bbox=[.08, .04, .92, .86])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_linewidth(0)
        if r == 0 or c == -1:
            cell.set_facecolor('white')
            cell.set_text_props(weight='bold', color='#354451')
        if r > 0 and c >= 0 and r - 1 == c:
            cell.set_facecolor('white')
            cell.set_text_props(color='#919AA2')


def sensitivity(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.20, top=.77, wspace=.18)
    teacher_rows = [r for r in rows if r['case'] == 'teacher']
    ymin = min(v for r in teacher_rows for v in r['action_values'].values())
    ymax = max(v for r in teacher_rows for v in r['action_values'].values())
    for ax, depth in zip(axes, [2, 4]):
        data = sorted([r for r in teacher_rows if r['depth'] == depth], key=lambda r: r['value'])
        assert [r['value'] for r in data] == list(range(21))
        for i, target in enumerate(['T1', 'T2', 'T3', 'T4']):
            ax.plot([r['value'] for r in data], [r['action_values'][target] for r in data],
                    color=TARGET_COLORS[i], label=target, lw=1.5)
        ax.set_title(f'$H={depth}$', pad=8)
        ax.set(xlim=(-.4, 20.4), ylim=(ymin-1.5, ymax+2), xlabel='T3的价值 $V_3$')
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axes[0].set_ylabel('首步回传值 $U$')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.54, 1.005), frameon=False, ncol=4,
               handlelength=1.8, columnspacing=1.7)
    save(fig, 'q4_sensitivity_report')


def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    teacher = json.loads((DATA / 'teacher.json').read_text(encoding='utf-8'))
    sweeps = json.loads((DATA / 'value_sweeps.json').read_text(encoding='utf-8'))
    assert teacher['by_depth']['2']['action_values'] == dict(zip(['T1','T2','T3','T4'], [-4,-1,-2,-4]))
    assert teacher['by_depth']['4']['action_values'] == dict(zip(['T1','T2','T3','T4'], [-5,1,3,-7]))

    fig, ax = plt.subplots(figsize=(4.1, 2.95))
    fig.subplots_adjust(left=.15, bottom=.19, right=.99, top=.98)
    bars(ax, teacher)
    save(fig, 'q4_first_move_report')

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95),
                             gridspec_kw={'width_ratios': [1, 1.4]})
    fig.subplots_adjust(left=.045, right=.985, top=.88, bottom=.19, wspace=.38)
    matrix(axes[0], teacher)
    bars(axes[1], teacher)
    axes[0].set_title('（a）最小移动代价矩阵', pad=7)
    axes[1].set_title('（b）候选首步的回传值', pad=7)
    save(fig, 'q4_basis_report_layout')

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    fig.subplots_adjust(left=.09, right=.98, bottom=.02, top=.99)
    matrix(ax, teacher)
    save(fig, 'q4_distance_matrix_report')
    sensitivity(sweeps)
    provenance = {
        'script': 'scripts/plot_report_revision.py',
        'sources': ['outputs/q4/diagnostics/teacher.json', 'outputs/q4/diagnostics/value_sweeps.json'],
        'files': ['q4_first_move_report.png', 'q4_basis_report_layout.png',
                  'q4_distance_matrix_report.png', 'q4_sensitivity_report.png'],
        'note': '普通柱状图和折线图；不加最优标记、阈值线、网格或说明文字。仅重绘已保存数据。',
    }
    (OUT / 'q4_report_revision_manifest.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
