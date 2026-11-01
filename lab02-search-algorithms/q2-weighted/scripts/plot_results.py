# 普通 matplotlib 绘图；只输出 PNG，不使用绘图技能或模板包。
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from searchlab.fixtures import MAZE, WEIGHTED, teacher_tree
from searchlab.generators import load_tree, structure_grid
from searchlab.grid import Grid, astar, build_heuristic
from searchlab.tree import exact_values, search_tree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
FIG = OUT/'figures'
COLORS = ['#17658A', '#D27A22', '#38856A', '#9C4972', '#60666E', '#715CA7']
MARKERS = ['o','s','^','D','v','P']
LABEL = {'zero':'h=0','euclidean':'欧氏距离','manhattan':'曼哈顿距离','corrected':'修正曼哈顿','relaxed':'放松地图下界',
         'larger_g':'较大 g','smaller_g':'较小 g','fifo':'先入队', 'fixed':'原顺序','reverse':'反向','random':'随机',
         'oracle_good':'理想顺序','oracle_bad':'不利顺序','MM':'Minimax','MM+TT':'Minimax+缓存','AB':'Alpha-beta','AB+TT':'Alpha-beta+缓存'}
MANIFEST = []


def setup():
    font_manager.fontManager.addfont(r'C:\Windows\Fonts\msyh.ttc') if Path(r'C:\Windows\Fonts\msyh.ttc').exists() else None
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((name for name in ['Microsoft YaHei','Noto Sans CJK SC','SimHei','DejaVu Sans'] if name in available),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
                         'legend.fontsize':8,'figure.dpi':140,'savefig.dpi':300,'lines.linewidth':1.6,
                         'axes.axisbelow':True,'grid.alpha':.15})
    FIG.mkdir(parents=True,exist_ok=True)


def save(fig,name,claim,source):
    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    MANIFEST.append({'id':name,'path':str((FIG/(name+'.png')).relative_to(ROOT)),
                     'claim':claim,'source':source,'script':'scripts/plot_results.py'})


def read_rows():
    rows=[]
    for path in sorted((OUT/'raw').glob('Q[123]-*.jsonl')):
        rows.extend(json.loads(s) for s in path.read_text(encoding='utf-8').splitlines())
    return pd.DataFrame(rows)


def interval(ax,x,arrays,label,color,marker='o',linestyle='-'):
    med=np.array([np.median(a) for a in arrays])
    low=np.array([np.quantile(a,.25) for a in arrays])
    high=np.array([np.quantile(a,.75) for a in arrays])
    ax.plot(x,med,marker=marker,color=color,label=label,ls=linestyle,markersize=4)
    ax.fill_between(x,low,high,color=color,alpha=.12)


def draw_grid(ax,grid,path=None,title='',expanded=None,annotate=False):
    image=np.array([[0 if v is None else 1 for v in row] for row in grid.costs])
    ax.imshow(image,cmap=ListedColormap(['#28333E','#F4F4EE']),vmin=0,vmax=1,interpolation='nearest')
    if expanded:
        ax.scatter([p[1] for p in expanded],[p[0] for p in expanded],marker='s',s=9,
                   c=COLORS[0],alpha=.3,lw=0,zorder=2)
    if path:
        ax.plot([p[1] for p in path],[p[0] for p in path],'-',c=COLORS[1],lw=1.7,zorder=3)
    ax.scatter([grid.start[1]],[grid.start[0]],marker='o',c=COLORS[2],s=28,zorder=4,edgecolor='white',lw=.5)
    ax.scatter([grid.goal[1]],[grid.goal[0]],marker='*',c=COLORS[3],s=55,zorder=4,edgecolor='white',lw=.5)
    if annotate:
        for r,row in enumerate(grid.costs):
            for c,v in enumerate(row):
                if v is not None:
                    text='S' if (r,c)==grid.start else 'E' if (r,c)==grid.goal else str(v)
                    ax.text(c,r,text,ha='center',va='center',fontsize=8,zorder=5)
    ax.set_title(title,loc='left',pad=7)
    ax.set_xlabel('列坐标');ax.set_ylabel('行坐标')
    if grid.rows<=8:
        ax.set_xticks(range(grid.cols));ax.set_yticks(range(grid.rows))


def q1_figures(df):
    # 样例搜索区域由实际算法生成，并保存到结果目录。
    grid=Grid(MAZE,(0,0),(3,4),'maze')
    examples={}
    fig,axes=plt.subplots(1,2,figsize=(4.2,2.25),layout='constrained')
    for ax,kind in zip(axes,['zero','manhattan']):
        result=astar(grid,heuristic=kind,record=True)
        examples[kind]=result.to_dict()
        draw_grid(ax,grid,result.path,f'{LABEL[kind]}：扩展 {result.expanded} 个节点',result.expansion_order)
    fig.suptitle('原题：路径均为 7 步；启发函数改变搜索范围',fontsize=11)
    save(fig,'q1_teacher','相同最短路径，搜索范围不同','teacher instance + actual A* trace')
    (OUT/'examples').mkdir(exist_ok=True)
    (OUT/'examples/q1_teacher.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')

    data=df[(df.experiment_id=='Q1-01') & (df.instance_name!='teacher')]
    fig=plt.figure(figsize=(4.25,4.3),layout='constrained')
    layout=fig.add_gridspec(2,2)
    axes=[fig.add_subplot(layout[0,0]),fig.add_subplot(layout[0,1]),fig.add_subplot(layout[1,:])]
    for i,kind in enumerate(['zero','euclidean','manhattan']):
        sub=data[data.heuristic==kind]
        for ax,metric,scale in [(axes[0],'expanded',1),(axes[1],'elapsed_ns',1e-6)]:
            interval(ax,[15,30],[sub[sub['size']==s][metric].to_numpy()*scale for s in [15,30]],LABEL[kind],COLORS[i],MARKERS[i])
    axes[0].set(xlabel='地图边长（格）',ylabel='扩展节点数（个）',title='a  扩展节点数')
    axes[1].set(xlabel='地图边长（格）',ylabel='求解时间（ms）',title='b  实测时间')
    tie=df[(df.experiment_id=='Q1-02') & (df.instance_name=='open')]
    vals=[tie[tie.ordering==name].expanded.iloc[0] for name in ['larger_g','smaller_g','fifo']]
    axes[2].bar(range(3),vals,color=COLORS[:3],width=.6)
    for i,v in enumerate(vals):axes[2].text(i,v+max(vals)*.025,str(int(v)),ha='center',fontsize=8)
    axes[2].set_xticks(range(3),[LABEL[x] for x in ['larger_g','smaller_g','fifo']])
    axes[2].set(ylabel='扩展节点数（个）',title='c  开放图：相同 f 的不同选择',ylim=(0,max(vals)*1.17))
    axes[0].legend(frameon=False)
    fig.suptitle('a–b：每条件 N=20，点为中位数、阴影为四分位区间；c：确定性 50×50 地图',fontsize=9)
    save(fig,'q1_heuristics','启发函数与平局规则共同影响效率','Q1-01;Q1-02')

    fig,axes=plt.subplots(2,2,figsize=(4.3,4.5),layout='constrained')
    for ax,kind,label in zip(axes.flat,['open','serpentine','comb','detour'],['开放地图','蛇形走廊','梳状死路','远离目标绕行']):
        data,start,goal=structure_grid(kind)
        grid=Grid(data,start,goal,'maze')
        a=astar(grid,heuristic='manhattan',record=True);b=astar(grid,heuristic='zero')
        draw_grid(ax,grid,a.path,f'{label}\n扩展 {b.expanded} / {a.expanded}',a.expansion_order)
    fig.suptitle('确定性结构案例：橙线为路径，蓝色为曼哈顿 A* 的有效弹出区域',fontsize=10)
    save(fig,'q1_structures','地图结构决定启发函数能否缩小搜索','Q1-03 + saved generator definitions')

    data=df[(df.experiment_id=='Q1-04') & (df.heuristic=='manhattan')]
    fig,axes=plt.subplots(2,1,figsize=(4.2,4.6),layout='constrained')
    densities=[0,.15,.30,.40]
    for i,size in enumerate([15,30,50]):
        sub=data[data['size']==size]
        axes[0].plot(densities,[sub[sub.density==p].reachable.astype(float).mean() for p in densities],marker=MARKERS[i],color=COLORS[i],label=f'{size}×{size}')
    axes[0].set(xlabel='障碍概率',ylabel='样本可达比例',ylim=(-.03,1.04),title='a  不筛选不可达地图')
    axes[0].legend(frameon=False)
    sub=data[data['size']==50]
    for reachable,color,marker,label in [(True,COLORS[0],'o','可达'),(False,COLORS[1],'s','不可达')]:
        xs=[];arrays=[]
        for p in densities:
            arr=sub[(sub.density==p)&(sub.reachable==reachable)].expanded.to_numpy()
            if len(arr):xs.append(p);arrays.append(arr)
        interval(axes[1],xs,arrays,label,color,marker)
    axes[1].set(xlabel='障碍概率',ylabel='扩展节点数（个）',title='b  50×50：按可达性分层')
    axes[1].legend(frameon=False)
    fig.suptitle('随机条件 N=20；p=0 的重复地图去重为 N=1；阴影为条件内四分位区间',fontsize=9)
    save(fig,'q1_scale','高障碍下可达率与搜索开销必须分开解释','Q1-04')


def q2_figures(df):
    data=df[df.experiment_id=='Q2-02']
    optimal=data[data.heuristic=='corrected'].iloc[0]
    bfs=data[data.algorithm=='BFS'].iloc[0]
    payload=json.loads(gzip.decompress((OUT/'instances'/(optimal.instance_sha256+'.json.gz')).read_bytes()))
    grid=Grid(payload['grid'],payload['start'],payload['goal'],'weighted')
    fig,axes=plt.subplots(2,1,figsize=(4.2,4.6),layout='constrained')
    draw_grid(axes[0],grid,optimal.path,'a  绕行避开高代价地形',annotate=True)
    axes[0].plot([p[1] for p in bfs.path],[p[0] for p in bfs.path],ls='--',c=COLORS[3],lw=1.6,label='BFS 最少步数')
    axes[0].plot([],[],c=COLORS[1],label='A* 最小代价');axes[0].legend(loc='lower left',frameon=True,fontsize=7)
    for row,color,marker,label in [(bfs,COLORS[3],'s','BFS：6 步 / 代价 40'),(optimal,COLORS[0],'o','A*：8 步 / 代价 7')]:
        costs=[0];total=0
        for a,b in row.path[1:]:
            total+=grid.costs[a][b];costs.append(total)
        axes[1].plot(range(len(costs)),costs,marker=marker,color=color,label=label,markersize=3)
    axes[1].set(xlabel='已移动步数（步）',ylabel='累计进入代价',title='b  最后进入 E 的代价为 0')
    axes[1].legend(frameon=False,loc='upper left')
    fig.suptitle('确定性反例 N=1：最少步数与最小代价是不同目标',fontsize=10)
    save(fig,'q2_costs','短路线并不一定代价小','Q2-02')

    data=df[(df.experiment_id=='Q2-04') & (df['size']==50)]
    fig,axes=plt.subplots(2,1,figsize=(4.2,4.6),layout='constrained')
    counts=[1,10,50]
    for i,h in enumerate(['zero','corrected','relaxed']):
        sub=data[data.heuristic==h]
        interval(axes[0],counts,[sub[sub.query_count==n].expanded.to_numpy() for n in counts],LABEL[h],COLORS[i],MARKERS[i])
        interval(axes[1],counts,[sub[sub.query_count==n].total_elapsed_ns.to_numpy()*1e-6 for n in counts],LABEL[h],COLORS[i],MARKERS[i])
    axes[0].set(xlabel='同一终点的查询数 K',ylabel='总扩展节点数（个）',title='a  更强下界减少搜索')
    axes[1].set(xlabel='同一终点的查询数 K',ylabel='预处理＋K 次查询时间（ms）',title='b  将预处理成本计入比较')
    axes[0].legend(frameon=False)
    fig.suptitle('50×50 加权图；每条件 N=20；中位数及四分位区间',fontsize=10)
    save(fig,'q2_heuristics','强启发函数必须同时评估预处理与查询成本','Q2-04')

    data=df[df.experiment_id=='Q2-03']
    patterns=['uniform','mild','sparse','barrier','corridor']
    labels=['均匀','轻度差异','稀疏高价','高价屏障','低价走廊']
    fig,axes=plt.subplots(2,1,figsize=(4.2,4.6),layout='constrained')
    for i,h in enumerate(['zero','corrected']):
        sub=data[data.heuristic==h]
        arrays=[sub[sub.pattern==p].expanded.to_numpy() for p in patterns]
        med=np.array([np.median(a) for a in arrays]);lo=np.array([np.quantile(a,.25) for a in arrays]);hi=np.array([np.quantile(a,.75) for a in arrays])
        axes[0].bar(np.arange(5)+(i-.5)*.32,med,width=.32,color=COLORS[i],label=LABEL[h],yerr=np.array([med-lo,hi-med]),capsize=2,error_kw={'lw':.8})
    axes[0].set_xticks(range(5),labels,rotation=20);axes[0].set(ylabel='扩展节点数（个）',title='a  权重结构影响搜索量');axes[0].legend(frameon=False)
    sub=data[data.heuristic=='corrected']
    arrays=[sub[sub.pattern==p]['cost'].to_numpy() for p in patterns]
    values=np.array([np.median(a) for a in arrays]);low=np.array([np.quantile(a,.25) for a in arrays]);high=np.array([np.quantile(a,.75) for a in arrays])
    axes[1].errorbar(range(5),values,yerr=[values-low,high-values],marker='s',color=COLORS[1],capsize=3,lw=1.5)
    axes[1].set_xticks(range(5),labels,rotation=20);axes[1].set(ylabel='最小路径代价',title='b  步数相同，代价不同')
    fig.suptitle('30×30；轻度差异/稀疏高价 N=20，其余 N=1；误差线为四分位区间',fontsize=9)
    save(fig,'q2_patterns','权重结构影响代价和搜索而非仅影响步数','Q2-03')


def q3_tree_figure():
    # 教材式博弈树：正三角MAX、倒三角MIN，空心节点和简洁叶值。
    # 将共享的max2子树展开两次，以树形方式展示两条父子关系。
    root=teacher_tree(); oracle=exact_values(root)
    mm=search_tree(root,pruning=False,record=True);ab=search_tree(root,record=True)
    fig,ax=plt.subplots(figsize=(4.6,2.45),layout='constrained')
    positions={(): (3.5,3)}
    for depth in range(1,4):
        for i in range(2**depth):
            bits=tuple((i>>(depth-j-1))&1 for j in range(depth))
            positions[bits]=((i+.5)*8/(2**depth)-.5,3-depth)
    def draw(node,path=(),maximizing=True):
        x,y=positions[path]
        pruned=path==(0,1,1)
        if path:
            px,py=positions[path[:-1]]
            # 叶值不画边框，连接线在数字上方结束；三角形白底遮住内部连线。
            endpoint=(px+(x-px)*.82, py+(y-py)*.82) if not node.children else (x,y)
            ax.plot([px,endpoint[0]],[py,endpoint[1]],color='#B0B5BA' if pruned else '#5C636A',
                    lw=.85,ls=(0,(3,2)) if pruned else '-',zorder=1)
            if pruned:
                cx,cy=px+(x-px)*.57,py+(y-py)*.57
                ax.plot([cx-.07,cx+.07],[cy-.07,cy+.07],color='#8A8F94',lw=1,zorder=3)
                ax.plot([cx-.07,cx+.07],[cy+.07,cy-.07],color='#8A8F94',lw=1,zorder=3)
        value=oracle[id(node),maximizing]
        text=str(value)
        if path==(0,1):
            text='≥'+str(ab.visited_leaves[2])
        if node.children:
            half_width=.30
            height=np.sqrt(3)*half_width
            vertices=[(x-half_width,y-height/3),(x+half_width,y-height/3),(x,y+2*height/3)]
            if not maximizing:
                vertices=[(vx,2*y-vy) for vx,vy in vertices]
            ax.add_patch(Polygon(vertices,closed=True,facecolor='white',edgecolor='#20262C',lw=1,zorder=2))
            text_y=y-.025 if maximizing else y+.025
            ax.text(x,text_y,text,ha='center',va='center',fontsize=9,color='#171B20',zorder=3)
        else:
            ax.text(x,y,text,ha='center',va='center',fontsize=10,color='#A9AEB3' if pruned else '#171B20',zorder=3)
        if pruned:ax.text(x,y-.27,'剪去',ha='center',va='top',fontsize=7,color='#8A8F94')
        for i,child in enumerate(node.children):draw(child,path+(i,),not maximizing)
    draw(root)
    for y,label in [(3,'MAX'),(2,'MIN'),(1,'MAX')]:
        ax.text(-.65,y,label,ha='right',va='center',fontsize=7.5,color='#59616A')
    ax.set(xlim=(-1.25,7.5),ylim=(-.55,3.48),aspect='equal');ax.axis('off')
    save(fig,'q3_tree','用实际回传值解释原题的一次有效剪枝','teacher tree + actual search counts')
    (OUT/'examples').mkdir(exist_ok=True)
    (OUT/'examples/q3_teacher.json').write_text(json.dumps({'minimax':mm.to_dict(),'alphabeta':ab.to_dict()},ensure_ascii=False,indent=2),encoding='utf-8')


def q3_figures(df):
    q3_tree_figure()
    a=df[(df.experiment_id=='Q3-01') & (df.instance_name=='random_b3_d6')]
    b=df[(df.experiment_id=='Q3-02') & (df.branching==3)&(df.depth==6)]
    arrays=[]; labels=[]
    for alg,order,frame,label in [('MM','fixed',a,'Minimax'),('AB','fixed',a,'原顺序'),('AB','reverse',a,'反向'),('AB','random',a,'随机'),('AB','oracle_good',b,'理想'),('AB','oracle_bad',b,'不利')]:
        arrays.append(frame[(frame.algorithm==alg)&(frame.ordering==order)].calls.to_numpy());labels.append(label)
    fig,ax=plt.subplots(figsize=(4.25,2.8),layout='constrained')
    box=ax.boxplot(arrays,tick_labels=labels,patch_artist=True,showfliers=True,widths=.55,
                   medianprops={'color':'#142C3A','linewidth':1.3})
    for i,patch in enumerate(box['boxes']):patch.set(facecolor=COLORS[i],alpha=.38)
    ax.set(ylabel='递归访问次数（含根与叶）',title='3 叉 / 深度 6：不同访问顺序')
    fig.suptitle('每条件 N=20；箱体为四分位区间；理想/不利顺序使用预知子树值',fontsize=9)
    save(fig,'q3_ordering','访问顺序决定Alpha-beta收益的幅度','Q3-01;Q3-02')

    data=df[df.experiment_id=='Q3-03']
    fig,axes=plt.subplots(2,1,figsize=(4.2,4.6),layout='constrained')
    for ax,p,title in [(axes[0],0,'a  均衡树'),(axes[1],.2,'b  非均衡树：提前终止概率 0.2')]:
        for i,b in enumerate([2,3,4]):
            for alg,ls,marker in [('MM','--','s'),('AB','-','o')]:
                sub=data[(data.early_stop==p)&(data.branching==b)&(data.algorithm==alg)]
                interval(ax,[4,6,8],[sub[sub.depth==d].calls.to_numpy() for d in [4,6,8]],f'b={b} {alg}',COLORS[i],marker,ls)
        ax.set(xlabel='树深度（边）',ylabel='递归访问次数（对数轴）',yscale='log',title=title)
    axes[0].legend(frameon=False,fontsize=7,ncol=2)
    fig.suptitle('每条件 N=20；中位数及四分位区间；不同结构使用其实际节点数',fontsize=9)
    save(fig,'q3_scale','剪枝减少访问，但分支和深度增长仍是主要瓶颈','Q3-03')

    data=df[df.experiment_id=='Q3-04']
    fig,axes=plt.subplots(2,1,figsize=(4.2,4.6),layout='constrained')
    ps=[0,.25,.5,.75]
    for i,alg in enumerate(['MM+TT','AB','AB+TT']):
        arrays_calls=[];arrays_time=[]
        for p in ps:
            sub=data[data.reuse_probability==p]
            base=sub[sub.algorithm=='MM'].set_index('seed')
            trial=sub[sub.algorithm==alg].set_index('seed').loc[base.index]
            arrays_calls.append((trial.calls/base.calls).to_numpy())
            arrays_time.append((trial.elapsed_ns/base.elapsed_ns).to_numpy())
        interval(axes[0],ps,arrays_calls,LABEL[alg],COLORS[i],MARKERS[i])
        interval(axes[1],ps,arrays_time,LABEL[alg],COLORS[i],MARKERS[i])
    for ax in axes:ax.axhline(1,color='#777777',ls=':',lw=1);ax.set_xlabel('生成时的节点复用概率 p')
    axes[0].set(ylabel='访问次数 / 无缓存 Minimax',title='a  减少重复求值')
    axes[1].set(ylabel='耗时 / 无缓存 Minimax',title='b  检验缓存自身的开销')
    axes[0].legend(frameon=False,fontsize=7)
    fig.suptitle('3 叉 / 深度 8；每条件 N=20；配对比值中位数与四分位区间',fontsize=9)
    save(fig,'q3_cache','缓存收益依赖真实重复状态及查询开销','Q3-04')


def main():
    setup()
    df=read_rows()
    expected={f'Q{q}-{i:02d}' for q in range(1,4) for i in range(1,5)}
    if not expected.issubset(set(df.experiment_id)):
        raise ValueError('run all 12 experiment groups before generating final figures')
    q1_figures(df);q2_figures(df);q3_figures(df)
    with (FIG/'figure_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','path','claim','source','script']);writer.writeheader();writer.writerows(MANIFEST)
    print(f'Created {len(MANIFEST)} PNG figures in {FIG}')


if __name__=='__main__':main()
