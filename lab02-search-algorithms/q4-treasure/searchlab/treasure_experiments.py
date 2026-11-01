# 第四题五组正式实验；与前三题数据隔离运行，可逐条恢复。
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
import tracemalloc

from .generators import save_instance
from .treasure import TreasureProblem, build_distances, diagnose_game, search_game
from .treasure_generators import random_case, scaled_cases, subset_case
from .treasure_reference import reference_distances, reference_game

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
QOUT = OUT/'q4'
IDS = [f'Q4-{i:02d}' for i in range(1, 6)]
SEEDS = range(10, 30)


def source_hash():
    names = ['grid.py', 'generators.py', 'reference.py', 'treasure.py',
             'treasure_reference.py', 'treasure_generators.py', 'treasure_experiments.py']
    return hashlib.sha256(b''.join((Path(__file__).parent/n).read_bytes() for n in names)).hexdigest()


def curated_cases():
    return {c['id']: c for c in json.loads((ROOT/'设计材料/第四题案例定义.json').read_text(encoding='utf-8'))['cases']}


def measure(fn, timing='full'):
    def signature(value):
        return (value.best_target, value.best_value, value.status) if hasattr(value, 'best_target') else value.matrix
    start = perf_counter_ns()
    first = fn()
    probe = perf_counter_ns()-start
    if timing == 'single' or getattr(first, 'status', 'ok') != 'ok':
        return first, [probe], 1
    expected = signature(first)
    if probe >= 100_000_000:
        times = []
        for _ in range(3):
            start = perf_counter_ns()
            last = fn()
            times.append(perf_counter_ns()-start)
            if signature(last) != expected:
                raise AssertionError('repeated solve changed its result')
        return first, times, 1
    batch = max(1, min(20000, math.ceil(100_000_000/max(probe, 1))))
    for _ in range(batch):
        fn()
    times = []
    for _ in range(5):
        start = perf_counter_ns()
        for _ in range(batch):
            last = fn()
        times.append((perf_counter_ns()-start)/batch)
        if signature(last) != expected:
            raise AssertionError('repeated solve changed its result')
    return first, times, batch


class Runner:
    def __init__(self, selected, resume=False, timing='full'):
        self.selected, self.timing = selected, timing
        QOUT.mkdir(parents=True, exist_ok=True)
        (QOUT/'distances').mkdir(exist_ok=True)
        (QOUT/'diagnostics').mkdir(exist_ok=True)
        (OUT/'raw').mkdir(exist_ok=True)
        self.rows, self.done, self.geometry, self.references = [], {}, {}, {}
        if not resume:
            for ident in selected:
                path = OUT/'raw'/f'{ident}.jsonl'
                if path.exists():
                    path.unlink()
        for path in sorted((OUT/'raw').glob('Q4-*.jsonl')):
            for line in path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                if row['source_hash'] != source_hash() or row['timing_mode'] != timing:
                    raise ValueError('Q4 source/timing changed; rerun affected groups without resume')
                self.rows.append(row)
                self.done[row['run_key']] = row
        self.new = 0

    def prepare(self, case):
        physical = dict(grid=case['grid'], points=case['points'], labels=['A', 'B']+list(case['values']))
        key = hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest()
        if key in self.geometry:
            return self.geometry[key]
        path = QOUT/'distances'/f'{key}.json'
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            if saved['source_hash'] != source_hash() or saved['timing_mode'] != self.timing:
                saved = None
        else:
            saved = None
        if saved is None:
            data, times, batch = measure(lambda: build_distances(case), self.timing)
            expected = reference_distances(case)
            if data.matrix != expected:
                raise AssertionError('A* distance matrix disagrees with independent Dijkstra')
            for name, route in data.paths.items():
                start, finish = name.split('->')
                assert route[0] == case['points'][start] and route[-1] == case['points'][finish]
                assert all(abs(u[0]-v[0])+abs(u[1]-v[1]) == 1 for u, v in zip(route, route[1:]))
                assert sum(case['grid'][r][c] for r, c in route[1:]) == expected[data.labels.index(start)][data.labels.index(finish)]
            saved = dict(geometry_sha256=key, source_hash=source_hash(), timing_mode=self.timing,
                         physical=physical, **asdict(data), timings_ns=times, batch_size=batch,
                         preprocess_ns=statistics.median(times), independently_verified=True)
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
        self.geometry[key] = saved
        return saved

    def solve(self, ident, case, depth, algorithm='AB', *, seed=None,
              pruning=True, cache=False, ordering='fixed', eta=0., meta=None):
        payload = {'kind': 'treasure', 'case': case}
        digest, _ = save_instance(OUT/'instances', payload)
        key = '|'.join(map(str, [ident, digest, depth, pruning, cache, ordering, eta]))
        if key in self.done:
            return self.done[key]
        distances = self.prepare(case)
        problem = TreasureProblem(case, distances['matrix'])
        reference_key = digest, depth
        if reference_key not in self.references:
            self.references[reference_key] = reference_game(problem.distances, problem.values, depth)
        ref = self.references[reference_key]
        first, times, batch = measure(lambda: search_game(problem, depth, pruning=pruning, cache=cache,
                                                         ordering=ordering, eta=eta), self.timing)
        if first.status == 'ok':
            expected_target = problem.targets[min(ref['optimal_actions'])] if ref['optimal_actions'] else None
            if (first.best_value, first.best_target) != (ref['value'], expected_target):
                raise AssertionError((ident, case['id'], depth, algorithm, first.to_dict(), ref))
        else:
            assert first.best_value is None and first.best_target is None
        first.elapsed_ns = int(statistics.median(times))
        row = dict(experiment_id=ident, instance_name=case['id'], instance_sha256=digest,
                   geometry_sha256=distances['geometry_sha256'], source_hash=source_hash(),
                   seed=seed, size=len(case['grid']), treasure_count=problem.count, depth=depth,
                   algorithm=algorithm, pruning=pruning, cache=cache, ordering=ordering, eta=eta,
                   run_key=key, timing_mode=self.timing, timings_ns=times, batch_size=batch,
                   preprocess_ns=distances['preprocess_ns'],
                   end_to_end_ns=distances['preprocess_ns']+statistics.median(times),
                   distance_searches=distances['searches'], distance_expanded=distances['expanded'],
                   expected_value=ref['value'], optimal_actions=[problem.targets[t] for t in ref['optimal_actions']],
                   reference_states=ref['reference_states'], independently_verified=first.status == 'ok',
                   **(meta or {}), **asdict(first))
        with (OUT/'raw'/f'{ident}.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        self.rows.append(row)
        self.done[key] = row
        self.new += 1
        if self.new % 10 == 0:
            print(f'  {ident} new={self.new} instance={case["id"]} H={depth} {algorithm} {first.status}', flush=True)
        return row

    def flush(self):
        if not self.rows:
            return
        fields = sorted(set().union(*(row.keys() for row in self.rows)))
        with (QOUT/'results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
        summary = dict(rows=len(self.rows), counts={i: sum(r['experiment_id'] == i for r in self.rows) for i in IDS},
                       statuses={s: sum(r['status'] == s for r in self.rows) for s in sorted({r['status'] for r in self.rows})})
        (QOUT/'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


def tune_order(runner):
    path = QOUT/'ordering_selection.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['source_hash'] == source_hash() and saved['timing_mode'] == runner.timing:
            return saved
    print('TUNE ordering with development seeds 0..9 only', flush=True)
    rows = []
    for seed in range(10):
        case = random_case(15, 8, seed, 'ablation')
        dist = runner.prepare(case)
        problem = TreasureProblem(case, dist['matrix'])
        digest, _ = save_instance(OUT/'instances', {'kind': 'treasure', 'case': case})
        baseline = search_game(problem, 6)
        for eta in [0., .5, 1., 2.]:
            got, times, batch = measure(lambda: search_game(problem, 6, ordering='competition', eta=eta), runner.timing)
            assert got.best_value == baseline.best_value and got.best_target == baseline.best_target
            rows.append(dict(seed=seed, eta=eta, calls=got.calls, baseline_calls=baseline.calls,
                             calls_ratio=got.calls/baseline.calls, elapsed_ns=statistics.median(times),
                             timings_ns=times, batch_size=batch, instance_sha256=digest))
        print(f'  development seed {seed} done', flush=True)
    scores = []
    for eta in [0., .5, 1., 2.]:
        sub = [row for row in rows if row['eta'] == eta]
        scores.append(dict(eta=eta, median_calls_ratio=statistics.median(row['calls_ratio'] for row in sub),
                           median_elapsed_ns=statistics.median(row['elapsed_ns'] for row in sub)))
    rank = lambda row: (row['median_calls_ratio'], row['median_elapsed_ns'], row['eta'])
    best = min(scores, key=rank)
    positive = min((s for s in scores if s['eta'] > 0), key=rank)
    saved = dict(source_hash=source_hash(), timing_mode=runner.timing, development_seeds=list(range(10)),
                 test_seeds=list(SEEDS), selection_rule='median paired calls ratio, then time, then smaller eta',
                 selected_eta=best['eta'], positive_eta=positive['eta'], scores=scores, runs=rows)
    path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
    return saved


def modes(selection):
    eta = selection['selected_eta']
    return [('MM', False, False, 'fixed', 0.), ('AB', True, False, 'fixed', 0.),
            ('AB+Order', True, False, 'competition', eta), ('AB+TT', True, True, 'fixed', 0.),
            ('AB+Order+TT', True, True, 'competition', eta)]


def q401(runner, cases):
    case = cases['teacher']
    problem = TreasureProblem(case, runner.prepare(case)['matrix'])
    diagnostics = {}
    for depth in [1, 2, 3, 4]:
        for name, pruning in [('MM', False), ('AB', True)]:
            runner.solve('Q4-01', case, depth, name, pruning=pruning)
        diagnostics[str(depth)] = diagnose_game(problem, depth)
    (QOUT/'diagnostics/teacher.json').write_text(json.dumps(dict(case=case, **runner.prepare(case), by_depth=diagnostics), ensure_ascii=False, indent=2), encoding='utf-8')


def q402(runner, cases):
    diagnostics = {}
    for name in ['C1', 'C2', 'C3', 'C4']:
        case = cases[name]
        problem = TreasureProblem(case, runner.prepare(case)['matrix'])
        full = reference_game(problem.distances, problem.values, 4)
        diagnostic = dict(case=case, by_depth={})
        for depth, policy in [(1, 'greedy'), (2, 'shallow'), (4, 'full')]:
            row = runner.solve('Q4-02', case, depth, 'MM', pruning=False,
                               meta=dict(policy=policy))
            chosen = problem.targets.index(row['best_target'])
            diagnostic['by_depth'][str(depth)] = dict(**diagnose_game(problem, depth),
                                                      full_action_value=full['action_values'][chosen],
                                                      first_action_regret=full['value']-full['action_values'][chosen])
        # 完整轨迹用于单独解释固定贪心首步后的最优应对，不是双方持续贪心。
        diagnostics[name] = diagnostic
    (QOUT/'diagnostics/cases.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding='utf-8')


def q403(runner, selection):
    configs = modes(selection)
    for seed in SEEDS:
        case = random_case(15, 8, seed, 'ablation')
        for depth in [4, 6, 8]:
            shift = seed % len(configs)
            for name, pruning, cache, ordering, eta in configs[shift:]+configs[:shift]:
                runner.solve('Q4-03', case, depth, name, seed=seed, pruning=pruning, cache=cache,
                             ordering=ordering, eta=eta, meta=dict(analysis_role='ablation'))
        for order, eta, name in [('fixed', 0., 'AB'), ('immediate', 0., 'AB+Immediate'),
                                  ('competition', selection['positive_eta'], 'AB+Competition')]:
            # eta=0的竞争排序与即时收益排序相同，但如主版本已使用相同语义，不重复计时。
            canonical_order = 'competition' if order == 'immediate' and selection['selected_eta'] == 0 else order
            runner.solve('Q4-03', case, 6, name, seed=seed, ordering=canonical_order, eta=eta,
                         meta=dict(analysis_role='ordering'))
        gc.collect()


def q404(runner, cases):
    outputs = []
    for name, target, values in [('teacher', 'T3', range(21)), ('C3', 'T2', range(41))]:
        for value in values:
            base = cases[name]
            case = {**base, 'id': f'{name}_{target}_{value}', 'values': {**base['values'], target: value}}
            for depth in [2, 4]:
                for alg, pruning in [('MM', False), ('AB', True)]:
                    row = runner.solve('Q4-04', case, depth, alg, pruning=pruning,
                                       meta=dict(sweep_case=name, sweep_target=target, sweep_value=value))
                    if alg == 'MM':
                        problem = TreasureProblem(case, runner.prepare(case)['matrix'])
                        outputs.append(dict(case=name, target=target, value=value,
                                            **diagnose_game(problem, depth)))
    (QOUT/'diagnostics/value_sweeps.json').write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding='utf-8')


def scaling_inputs(seed):
    for case in scaled_cases(seed):
        yield case, 6, dict(axis='map_size', axis_value=len(case['grid']))
    base = random_case(30, 12, seed, 'treasure_count', (20, 60))
    for count in [4, 6, 8, 10, 12]:
        yield subset_case(base, count), 4, dict(axis='treasure_count', axis_value=count)
    base = random_case(30, 10, seed, 'depth', (20, 60))
    for depth in [2, 4, 6, 8]:
        yield base, depth, dict(axis='depth', axis_value=depth)
    for count in [4, 6, 8]:
        yield random_case(15, count, seed, 'full_search'), count, dict(axis='full_search_reference', axis_value=count)


def q405(runner, selection):
    configs = [m for m in modes(selection) if m[0] in ('MM', 'AB', 'AB+Order+TT')]
    for seed in SEEDS:
        for case, depth, meta in scaling_inputs(seed):
            shift = seed % len(configs)
            for name, pruning, cache, ordering, eta in configs[shift:]+configs[:shift]:
                runner.solve('Q4-05', case, depth, name, seed=seed, pruning=pruning, cache=cache,
                             ordering=ordering, eta=eta, meta=meta)
        gc.collect()


def memory_profiles(runner, selection):
    output = []
    for case, depth, meta in scaling_inputs(10):
        if (meta['axis'], meta['axis_value']) not in [('map_size', 50), ('treasure_count', 12), ('depth', 8)]:
            continue
        problem = TreasureProblem(case, runner.prepare(case)['matrix'])
        gc.collect()
        tracemalloc.start()
        result = search_game(problem, depth, cache=True, ordering='competition', eta=selection['selected_eta'])
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        output.append(dict(**meta, seed=10, depth=depth, algorithm='AB+Order+TT',
                           scope='Python allocations during a separate search; excludes existing map/matrix',
                           current_bytes=current, peak_bytes=peak, status=result.status,
                           excluded_from_timing_comparison=True))
    (QOUT/'memory_profiles.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')


def run(ids='q4', resume=False, timing='full'):
    selected = IDS if ids.lower() in ('q4', 'all', 'q4-all') else ids.split(',')
    if any(i not in IDS for i in selected):
        raise ValueError('Q4 experiment ids must be Q4-01..05')
    QOUT.mkdir(parents=True, exist_ok=True)
    runner = Runner(selected, resume, timing)
    env = dict(source_hash=source_hash(), python=sys.version, executable=sys.executable,
               platform=platform.platform(), processor=platform.processor(), threads=1,
               timer='perf_counter_ns', timing_mode=timing, test_seeds=list(SEEDS), development_seeds=list(range(10)),
               seconds_limit=60, calls_limit=10_000_000, tt_capacity=200000,
               micro_batch_target_ns=100_000_000, micro_timed_batches=5, larger_timed_repeats=3,
               reset_table_each_solve=True, counts_include_root=True,
               distances='all ordered pairs via A*; special-cell entering cost 1',
               main_search_timing='excludes independent references, diagnostics, distances and I/O; includes sorting/TT/tie re-search')
    (QOUT/'environment.json').write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding='utf-8')
    cases = curated_cases()
    selection = tune_order(runner) if any(i in selected for i in ['Q4-03', 'Q4-05']) else None
    if selection:
        print(f'FROZEN eta={selection["selected_eta"]}; best positive eta={selection["positive_eta"]}', flush=True)
    try:
        for ident in selected:
            print('START '+ident, flush=True)
            if ident == 'Q4-01': q401(runner, cases)
            elif ident == 'Q4-02': q402(runner, cases)
            elif ident == 'Q4-03': q403(runner, selection)
            elif ident == 'Q4-04': q404(runner, cases)
            else: q405(runner, selection)
            runner.flush()
            print('DONE '+ident, flush=True)
        if 'Q4-05' in selected:
            memory_profiles(runner, selection)
    finally:
        runner.flush()
    print(json.dumps(json.loads((QOUT/'run_summary.json').read_text(encoding='utf-8')), ensure_ascii=False), flush=True)
