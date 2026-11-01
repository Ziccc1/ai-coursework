# 前三题的 12 组实验：保留原始运行、实例、验证与计时数据。
from dataclasses import asdict
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns

from .fixtures import MAZE, WEIGHTED, teacher_tree
from .generators import (local_random, random_dag, random_grid, random_tree,
                         save_instance, structure_grid, tree_payload, weighted_grid)
from .grid import Grid, astar, build_heuristic
from .reference import bfs_route, check_route, reference_tree, shortest_distances
from .tree import exact_values, search_tree, validate_tree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
SEEDS = range(10, 30)
IDS = [f'Q{q}-{i:02d}' for q in range(1, 4) for i in range(1, 5)]


def measure(fn, timing):
    begin = perf_counter_ns()
    first = fn()
    first_ns = perf_counter_ns() - begin
    if timing == 'single' or getattr(first, 'status', 'ok') not in ('ok', 'unreachable'):
        return first, [first_ns], 1
    if first_ns >= 100_000_000:
        times = [first_ns]
        for _ in range(2):
            begin = perf_counter_ns()
            value = fn()
            times.append(perf_counter_ns() - begin)
            assert value.value == first.value if hasattr(first, 'value') else value.cost == first.cost
        return first, times, 1
    batch = max(1, min(20_000, math.ceil(100_000_000 / max(first_ns, 1))))
    # 先预热一次，每次求解均从空搜索缓存开始。
    for _ in range(batch):
        fn()
    times = []
    for _ in range(5):
        begin = perf_counter_ns()
        for _ in range(batch):
            fn()
        times.append((perf_counter_ns() - begin) / batch)
    return first, times, batch


class Runner:
    def __init__(self, ids, resume, timing):
        self.timing, self.ids = timing, ids
        self.directory = OUT / 'raw'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows, self.done = [], set()
        if resume:
            for path in sorted(self.directory.glob('Q[123]-*.jsonl')):
                for line in path.read_text(encoding='utf-8').splitlines():
                    row = json.loads(line)
                    self.rows.append(row)
                    self.done.add(row['run_key'])
        else:
            for ident in ids:
                path = self.directory / (ident + '.jsonl')
                if path.exists():
                    path.unlink()
            # 汇总时保留其他已完成实验组的结果。
            for path in sorted(self.directory.glob('Q[123]-*.jsonl')):
                if path.stem not in ids:
                    self.rows.extend(json.loads(s) for s in path.read_text(encoding='utf-8').splitlines())
        self.count = 0

    def emit(self, row):
        row['timing_mode'] = self.timing
        row['elapsed_ns'] = statistics.median(row['timings_ns'])
        with (self.directory / (row['experiment_id'] + '.jsonl')).open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.rows.append(row)
        self.done.add(row['run_key'])
        self.count += 1
        if self.count % 40 == 0:
            print(f"  {row['experiment_id']} total_new_runs={self.count}", flush=True)

    def grid(self, ident, name, data, start, goal, problem, seed, heuristics, ties=('larger_g',), meta=None, diagnostics=False):
        payload = {'kind': 'grid', 'grid': data, 'start': start, 'goal': goal, 'problem': problem}
        digest, _ = save_instance(OUT/'instances', payload)
        g = Grid(data, start, goal, problem)
        expected = shortest_distances(g.costs, g.start).get(g.goal, -1)
        truth = shortest_distances(g.costs, g.goal, reverse=True) if diagnostics else None
        for heuristic in heuristics:
            h, prep = build_heuristic(g, heuristic)
            bound = {}
            if diagnostics:
                traversable = [(r, c) for r, row in enumerate(g.costs) for c, v in enumerate(row) if v is not None]
                bound = {'h_goal': h(g.goal),
                         'overestimated_nodes': sum(h(n) > d for n, d in truth.items()),
                         'inconsistent_edges': sum(h(n) > w+h(v) for n in traversable for v, w in g.neighbors(n)),
                         'h_start': h(g.start)}
                if heuristic in ('zero', 'corrected', 'relaxed'):
                    assert bound['h_goal'] == 0 and bound['overestimated_nodes'] == 0 and bound['inconsistent_edges'] == 0
            for tie in ties:
                key = '|'.join(map(str, [ident, name, seed, heuristic, tie]))
                if key in self.done:
                    continue
                first, times, batch = measure(lambda: astar(g, heuristic=h, tie_break=tie), self.timing)
                if heuristic != 'manhattan' or problem == 'maze':
                    check_route(g, first, expected)
                row = dict(experiment_id=ident, instance_name=name, instance_sha256=digest, seed=seed,
                           algorithm='A*', heuristic=heuristic, ordering=tie, size=g.rows,
                           reachable=expected >= 0, expected_cost=expected, preprocess_ns=prep,
                           run_key=key, timings_ns=times, batch_size=batch,
                           **(meta or {}), **bound, **asdict(first))
                row['path_steps'] = len(first.path)-1 if first.path else -1
                row.pop('elapsed_ns')
                row.pop('expansion_order')
                self.emit(row)

    def tree(self, ident, name, root, seed, modes, meta=None):
        payload = tree_payload(root)
        digest, _ = save_instance(OUT/'instances', payload)
        unique = validate_tree(root)
        expected = reference_tree(root)
        begin = perf_counter_ns()
        oracle = exact_values(root) if any(order.startswith('oracle') for _, _, order in modes) else None
        prep = perf_counter_ns()-begin if oracle is not None else 0
        # 在不同种子下轮换算法运行顺序，减小缓存预热造成的偏差。
        shift = (seed or 0) % len(modes)
        modes = modes[shift:] + modes[:shift]
        for prune, cache, order in modes:
            algorithm = ('AB' if prune else 'MM') + ('+TT' if cache else '')
            key = '|'.join(map(str, [ident, name, seed, algorithm, order]))
            if key in self.done:
                continue
            first, times, batch = measure(lambda: search_tree(root, pruning=prune, cache=cache, ordering=order,
                                                             seed=seed or 0, oracle=oracle, validate=False), self.timing)
            if first.status == 'ok':
                assert first.value == expected, (ident, first.value, expected)
            row = dict(experiment_id=ident, instance_name=name, instance_sha256=digest, seed=seed,
                       algorithm=algorithm, ordering=order, expected_value=expected, unique_nodes=unique,
                       preprocess_ns=prep if order.startswith('oracle') else 0,
                       run_key=key, timings_ns=times, batch_size=batch, **(meta or {}), **asdict(first))
            row.pop('elapsed_ns')
            row.pop('visited_leaves')
            self.emit(row)
        del payload

    def flush(self):
        if not self.rows:
            return
        # CSV汇总全部指标，嵌套的诊断信息同时保存在JSONL中。
        fields = sorted(set().union(*(row.keys() for row in self.rows)))
        path = OUT / 'results.csv'
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        counts = {ident: sum(row['experiment_id'] == ident for row in self.rows) for ident in IDS}
        (OUT/'run_summary.json').write_text(json.dumps({'rows': len(self.rows), 'counts': counts,
                                                      'statuses': {status: sum(r['status'] == status for r in self.rows) for status in sorted(set(r['status'] for r in self.rows))}}, ensure_ascii=False, indent=2), encoding='utf-8')


def q1(r, ident):
    if ident == 'Q1-01':
        r.grid(ident, 'teacher', MAZE, (0,0), (3,4), 'maze', None, ['zero','euclidean','manhattan'])
        for size in [15,30]:
            for seed in SEEDS:
                data, attempts = random_grid(size,.20,seed,'Q1-01')
                r.grid(ident,f'random_{size}',data,(0,0),(size-1,size-1),'maze',seed,['zero','euclidean','manhattan'],meta={'density':.20,'attempts':attempts})
    elif ident == 'Q1-02':
        r.grid(ident,'open',[[1]*50 for _ in range(50)],(0,0),(49,49),'maze',None,['manhattan'],['larger_g','smaller_g','fifo'])
        for seed in SEEDS:
            data, attempts = random_grid(50,.20,seed,'Q1-02')
            r.grid(ident,'random_50',data,(0,0),(49,49),'maze',seed,['manhattan'],['larger_g','smaller_g','fifo'],meta={'density':.20,'attempts':attempts})
    elif ident == 'Q1-03':
        for name in ['open','serpentine','comb','detour']:
            data,start,goal=structure_grid(name)
            r.grid(ident,name,data,start,goal,'maze',None,['zero','euclidean','manhattan'])
    else:
        for size in [15,30,50]:
            for density in [0,.15,.30,.40]:
                for seed in ([10] if density==0 else SEEDS):
                    data,attempts=random_grid(size,density,seed,'Q1-04',connected=False)
                    r.grid(ident,f'random_{size}_{density}',data,(0,0),(size-1,size-1),'maze',seed,['zero','manhattan'],meta={'density':density,'attempts':attempts})


def q2(r, ident):
    if ident == 'Q2-01':
        for name,data,goal in [('adjacent',[[1,0]],(0,1)),('unit_line',[[1,1,0]],(0,2)),
                               ('zero_line',[[1,0,0]],(0,2)),('unreachable',[[1,None,0]],(0,2))]:
            r.grid(ident,name,data,(0,0),goal,'weighted',None,['zero','corrected','manhattan'],diagnostics=True)
        r.grid(ident,'teacher',WEIGHTED,(0,0),(4,4),'weighted',None,['zero','corrected','relaxed'],diagnostics=True)
    elif ident == 'Q2-02':
        data=[[1]*7 for _ in range(5)]
        for c in range(1,6): data[2][c]=8
        data[2][6]=0
        r.grid(ident,'short_vs_cheap',data,(2,0),(2,6),'weighted',None,['zero','corrected'])
        g=Grid(data,(2,0),(2,6),'weighted')
        route=bfs_route(g.costs,g.start,g.goal)
        key=ident+'|BFS'
        if key not in r.done:
            begin=perf_counter_ns(); path=bfs_route(g.costs,g.start,g.goal);elapsed=perf_counter_ns()-begin
            digest,_=save_instance(OUT/'instances',{'kind':'grid','grid':data,'start':g.start,'goal':g.goal,'problem':'weighted'})
            r.emit(dict(experiment_id=ident,instance_name='short_vs_cheap',instance_sha256=digest,seed=None,algorithm='BFS',heuristic='none',ordering='fifo',run_key=key,timings_ns=[elapsed],batch_size=1,status='ok',cost=sum(g.costs[a][b] for a,b in path[1:]),path=path,path_steps=len(path)-1,expected_cost=7,preprocess_ns=0,reachable=True,size=5))
    elif ident == 'Q2-03':
        for pattern in ['uniform','mild','sparse','barrier','corridor']:
            for seed in (SEEDS if pattern in ('mild','sparse') else [10]):
                data,attempts=weighted_grid(30,0,seed,pattern,'Q2-03')
                r.grid(ident,pattern,data,(0,0),(29,29),'weighted',seed,['zero','corrected'],meta={'pattern':pattern,'attempts':attempts})
    else:
        for size in [30,50]:
            for seed in SEEDS:
                data,attempts=weighted_grid(size,.20,seed,'mild','Q2-04')
                goal=size-1,size-1
                base=Grid(data,(0,0),goal,'weighted')
                truth=shortest_distances(base.costs,goal,reverse=True)
                starts=[n for n in truth if n!=goal and base.costs[n[0]][n[1]]==1]
                local_random('Q2-04_starts',seed,size=size).shuffle(starts)
                if len(starts)<50: raise RuntimeError('not enough starts')
                digest,_=save_instance(OUT/'instances',{'kind':'multiquery_grid','grid':data,'goal':goal,'starts':starts[:50],'problem':'weighted'})
                for heuristic in ['zero','corrected','relaxed']:
                    begin=perf_counter_ns(); h,prep=build_heuristic(base,heuristic);prep_total=perf_counter_ns()-begin
                    assert all(h(n)<=d for n,d in truth.items())
                    assert all(h(n)<=w+h(v) for n in truth for v,w in base.neighbors(n))
                    for count in [1,10,50]:
                        key='|'.join(map(str,[ident,size,seed,heuristic,count]))
                        if key in r.done: continue
                        grids=[Grid(data,start,goal,'weighted') for start in starts[:count]]
                        # 各次查询使用相同地形和固定终点，只改变起点。
                        def solve_all():
                            return [astar(g,heuristic=h) for g in grids]
                        # 将整组查询作为一个计时批次。
                        begin=perf_counter_ns(); first=solve_all(); single=perf_counter_ns()-begin
                        reps=5 if r.timing=='full' else 1
                        times=[single]
                        if r.timing=='full':
                            solve_all()
                            for _ in range(reps-1):
                                begin=perf_counter_ns();solve_all();times.append(perf_counter_ns()-begin)
                        for g,res in zip(grids,first): check_route(g,res,truth[g.start])
                        r.emit(dict(experiment_id=ident,instance_name=f'multiquery_{size}',instance_sha256=digest,seed=seed,algorithm='A*',heuristic=heuristic,ordering='larger_g',size=size,query_count=count,preprocess_ns=prep_total,run_key=key,timings_ns=times,batch_size=1,status='ok',expanded=sum(x.expanded for x in first),generated=sum(x.generated for x in first),query_costs=[x.cost for x in first],query_starts=starts[:count],total_elapsed_ns=statistics.median(times)+prep_total,attempts=attempts))


def q3(r, ident):
    fixed=[(False,False,'fixed'),(True,False,'fixed')]
    if ident=='Q3-01':
        root=teacher_tree()
        r.tree(ident,'teacher',root,None,[(False,False,'fixed'),(True,False,'fixed'),(True,False,'reverse')])
        for seed in SEEDS:
            r.tree(ident,'teacher_permutations',root,seed,[(True,False,'random')])
            r.tree(ident,'random_b3_d6',random_tree(3,6,seed),seed,[(False,False,'fixed'),(True,False,'fixed'),(True,False,'reverse'),(True,False,'random')],meta={'branching':3,'depth':6})
    elif ident=='Q3-02':
        for b in [2,3]:
            for d in [4,6,8]:
                for seed in SEEDS:
                    r.tree(ident,f'b{b}_d{d}',random_tree(b,d,seed),seed,[(False,False,'fixed'),(True,False,'fixed'),(True,False,'oracle_good'),(True,False,'oracle_bad')],meta={'branching':b,'depth':d})
    elif ident=='Q3-03':
        for b in [2,3,4]:
            for d in [4,6,8]:
                for p in [0,.20]:
                    for seed in SEEDS:
                        r.tree(ident,f'b{b}_d{d}_p{p}',random_tree(b,d,seed,p),seed,fixed,meta={'branching':b,'depth':d,'early_stop':p})
                    gc.collect()
    else:
        for p in [0,.25,.5,.75]:
            for seed in SEEDS:
                root=random_dag(3,8,seed,p)
                r.tree(ident,f'reuse_{p}',root,seed,[(False,False,'fixed'),(False,True,'fixed'),(True,False,'fixed'),(True,True,'fixed')],meta={'branching':3,'depth':8,'reuse_probability':p})


def run(ids='all', resume=False, timing='full'):
    selected=IDS if ids=='all' else ids.split(',')
    if any(ident not in IDS for ident in selected):
        raise ValueError('only Q1-01..04, Q2-01..04, Q3-01..04 are implemented')
    OUT.mkdir(exist_ok=True)
    metadata={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'processor':platform.processor(),'timer':'time.perf_counter_ns','seeds':list(SEEDS),'timing':timing,'threads':1,'limits':{'seconds':60,'calls_or_expansions':10_000_000},'source_hash':hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()}
    (OUT/'environment.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    runner=Runner(selected,resume,timing)
    try:
        for ident in selected:
            print('START '+ident,flush=True)
            (q1 if ident.startswith('Q1') else q2 if ident.startswith('Q2') else q3)(runner,ident)
            runner.flush()
            print('DONE '+ident,flush=True)
    finally:
        runner.flush()
    print(json.dumps(json.loads((OUT/'run_summary.json').read_text(encoding='utf-8')),ensure_ascii=False),flush=True)
