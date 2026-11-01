# 给定迷宫的数字标注图；普通Matplotlib，仅输出PNG，不改动被测A*。
import heapq
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle, Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from searchlab.grid import Grid, astar, build_heuristic


def record_steps(grid, kind):
    # 仅为画图重放堆状态，并与原算法的全部有效出队顺序交叉检查。
    h, _ = build_heuristic(grid, kind)
    actual = astar(grid, heuristic=kind, record=True)
    best, parent, queue = {grid.start: 0}, {}, [(h(grid.start), 0, 0, 0, grid.start)]
    sequence, steps = 0, []
    while queue:
        f, _, _, g, node = heapq.heappop(queue)
        if g != best[node]:
            continue
        if node != grid.goal:
            for nxt, cost in grid.neighbors(node):
                candidate = g + cost
                if candidate < best.get(nxt, float('inf')):
                    best[nxt], parent[nxt] = candidate, node
                    sequence += 1
                    heapq.heappush(queue, (candidate+h(nxt), -candidate, sequence, candidate, nxt))
        candidates = sorted((f0, neg_g, seq, g0, n) for f0, neg_g, seq, g0, n in queue if g0 == best[n])
        steps.append(dict(order=len(steps)+1, node=list(node), g=g, h=h(node), f=f,
                          open=[dict(node=list(n), g=g0, h=h(n), f=f0) for f0, _, _, g0, n in candidates]))
        if node == grid.goal:
            break
    assert [tuple(s['node']) for s in steps] == actual.expansion_order
    assert len(steps)-1 == actual.expanded
    route, node = [grid.goal], grid.goal
    while node != grid.start:
        node = parent[node]
        route.append(node)
    assert route[::-1] == actual.path
    assert sequence+1 == actual.generated
    return actual, best, h, steps


def draw(ax, grid, actual, best, h, steps, title):
    visited = {tuple(s['node']): s['order'] for s in steps}
    path_color, line_color, open_color = '#FFF0DD', '#C77424', '#E1EFF6'
    for r in range(grid.rows):
        for c in range(grid.cols):
            node = r, c
            blocked = grid.costs[r][c] is None
            face = '#29343E' if blocked else path_color if node in visited else open_color if node in best else '#FFFFFF'
            ax.add_patch(Rectangle((c-.5, r-.5), 1, 1, facecolor=face, edgecolor='#CBD2D5', linewidth=.75, zorder=1))
            if node in best:
                ax.text(c+.07, r+.16, f'{best[node]} / {int(h(node))} / {int(best[node]+h(node))}',
                        ha='center', va='center', fontsize=9.4, color='#27323C', zorder=5)
            if node in visited:
                px, py = c-.27, r-.27
                ax.add_patch(Circle((px, py), .128, facecolor='white', edgecolor=line_color, linewidth=1.1, zorder=6))
                ax.text(px, py, str(visited[node]), fontsize=8.5, ha='center', va='center', color='#8C4B0E', zorder=7)
                if node in (grid.start, grid.goal):
                    ax.text(c+.30, r-.10, 'S' if node == grid.start else 'E', ha='center', va='center',
                            fontsize=10, fontweight='bold', color='#8C4B0E', zorder=7)
    # 箭头沿格子上侧/左侧连接，避开中心的g/h/f数值。
    for (r0,c0),(r1,c1) in zip(actual.path, actual.path[1:]):
        if c0 == c1:
            destination, origin = (c1-.43,r1-.32), (c0-.43,r0-.15)
        else:
            destination, origin = (c1-.27,r1-.27), (c0-.27,r0-.27)
        ax.annotate('', xy=destination, xytext=origin,
                    arrowprops=dict(arrowstyle='-|>', color=line_color, lw=1.7,
                                    shrinkA=8, shrinkB=8, mutation_scale=10), zorder=4)
    ax.set(xlim=(-.5,grid.cols-.5), ylim=(grid.rows-.5,-.5), aspect='equal',
           xticks=range(grid.cols), yticks=range(grid.rows), xlabel='列坐标 c', ylabel='行坐标 r')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=11)
    ax.tick_params(length=0, pad=5, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#89949C'); spine.set_linewidth(.8)


def main():
    for path in [Path('C:/Windows/Fonts/msyh.ttc')]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((n for n in ['Microsoft YaHei','Noto Sans CJK SC','SimHei'] if n in available), 'DejaVu Sans')
    plt.rcParams.update({'font.family':font, 'axes.unicode_minus':False, 'font.size':10})
    data = json.loads((ROOT/'inputs/q1_teacher.json').read_text(encoding='utf-8-sig'))
    grid = Grid(data['grid'], data['start'], data['goal'], 'maze')
    traces = {kind: record_steps(grid, kind) for kind in ['zero','manhattan']}
    figure_dir, example_dir = ROOT/'outputs/figures', ROOT/'outputs/examples'
    figure_dir.mkdir(parents=True, exist_ok=True)
    example_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.9,6.2))
    fig.subplots_adjust(left=.14,right=.96,top=.88,bottom=.24)
    draw(ax,grid,*traces['manhattan'],'')
    fig.suptitle('曼哈顿距离：7次扩展，路径7步',x=.55,y=.985,fontsize=12,fontweight='bold')
    fig.text(.55,.94,'格内数值：g / h / f；圆圈：有效出队序号',ha='center',fontsize=10)
    handles = [Patch(facecolor='#FFF0DD',edgecolor='#CBD2D5',label='已出队'),
               Patch(facecolor='#E1EFF6',edgecolor='#CBD2D5',label='仍在OPEN'),
               Patch(facecolor='white',edgecolor='#CBD2D5',label='尚未入队')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.55,.125),ncol=3,frameon=False,fontsize=9,handlelength=1.1,columnspacing=1)
    fig.text(.55,.076,'优先取最小 f；f 相同取较大 g；仍相同按入队先后',ha='center',fontsize=9)
    fig.text(.55,.043,'终点为第8次有效出队，直接返回，不计入扩展节点数',ha='center',fontsize=8.8,color='#515C65')
    out = figure_dir/'q1_teacher_trace.png'
    fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    # 对比版纵向排列，保持缩至报告单栏时数值仍可辨认。
    fig, axes = plt.subplots(2,1,figsize=(4.9,10.0))
    fig.subplots_adjust(left=.15,right=.96,top=.95,bottom=.08,hspace=.30)
    for ax,kind,title in zip(axes,['zero','manhattan'],['零启发：15次扩展','曼哈顿距离：7次扩展']):
        draw(ax,grid,*traces[kind],title)
    fig.text(.55,.026,'格内数值：g / h / f；圆圈为有效出队序号（含终点）',ha='center',fontsize=9)
    fig.savefig(figure_dir/'q1_teacher_trace_comparison.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    records = {kind:dict(steps=values[3], path=values[0].path, expanded=values[0].expanded,
                         note='可视化重放与原A*的完整有效出队顺序、路径、生成数逐项一致') for kind,values in traces.items()}
    (example_dir/'q1_teacher_trace.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out)
    print(json.dumps({k:[(s['order'],s['node'],s['g'],s['h'],s['f']) for s in v[3]] for k,v in traces.items()},ensure_ascii=False))


if __name__ == '__main__':
    main()
