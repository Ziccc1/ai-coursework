# 第四题逐条独立校验、覆盖检查，并合并四题结果索引。
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

from searchlab.treasure_experiments import IDS, source_hash
from searchlab.treasure_reference import enumerate_game, reference_distances, reference_game

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
QOUT = OUT/'q4'
EXPECTED = {'Q4-01': 8, 'Q4-02': 12, 'Q4-03': 320, 'Q4-04': 248, 'Q4-05': 960}


def measured_version(digest):
    # 注释修订前后的源码版本有明确对应；正式记录仍保留原校验值。
    revision = json.loads((ROOT/'source_revision.json').read_text(encoding='utf-8'))
    for relative, item in revision['files'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == item['after_sha256'], relative
    return revision['source_hash_aliases'].get(digest, digest)


def verify_trace(line, matrix, case, expected, fixed_first=False):
    labels = ['A', 'B']+sorted(case['values'], key=lambda t: int(t[1:]))
    values = [case['values'][t] for t in labels[2:]]
    positions, scores = [0, 1], [0, 0]
    mask = (1 << len(values))-1
    for index, event in enumerate(line):
        who = index % 2
        target = labels.index(event['target'])-2
        assert mask & (1 << target)
        if index != 0 or not fixed_first:
            reference = reference_game(matrix, values, len(line)-index,
                                       a=positions[0], b=positions[1], mask=mask, player=who)
            assert target == min(reference['optimal_actions'])
        distance = matrix[positions[who]][target+2]
        gain = values[target]-distance
        assert event['origin'] == labels[positions[who]]
        assert event['distance'] == distance and event['value'] == values[target]
        assert event['gain'] == gain and event['turn'] == index+1
        assert event['player'] == ('A' if who == 0 else 'B')
        positions[who] = target+2
        scores[who] += gain
        mask ^= 1 << target
        assert (event['score_a'], event['score_b'], event['utility']) == (scores[0], scores[1], scores[0]-scores[1])
    assert scores[0]-scores[1] == expected


def main():
    rows, seen, cases, distances, refs = [], set(), {}, {}, {}
    current_hash = measured_version(source_hash())
    selection = json.loads((QOUT/'ordering_selection.json').read_text(encoding='utf-8'))
    assert selection['source_hash'] == current_hash
    assert set(selection['development_seeds']).isdisjoint(selection['test_seeds'])
    assert selection['development_seeds'] == list(range(10))
    assert selection['test_seeds'] == list(range(10, 30))
    development_hashes = {r['instance_sha256'] for r in selection['runs']}
    for ident in IDS:
        group = [json.loads(s) for s in (OUT/'raw'/f'{ident}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(group) == EXPECTED[ident], (ident, len(group))
        for row in group:
            assert row['source_hash'] == current_hash
            assert row['run_key'] not in seen
            seen.add(row['run_key'])
            assert row['timing_mode'] == 'full'
            assert row['status'] in ('ok', 'timeout', 'node_limit')
            assert all(t > 0 for t in row['timings_ns'])
            assert row['elapsed_ns'] > 0 and row['batch_size'] >= 1
            digest = row['instance_sha256']
            assert digest not in development_hashes
            if digest not in cases:
                raw = gzip.decompress((OUT/'instances'/f'{digest}.json.gz').read_bytes())
                assert hashlib.sha256(raw).hexdigest() == digest
                cases[digest] = json.loads(raw)['case']
            case = cases[digest]
            geom = row['geometry_sha256']
            if geom not in distances:
                saved = json.loads((QOUT/'distances'/f'{geom}.json').read_text(encoding='utf-8'))
                assert saved['source_hash'] == current_hash and saved['timing_mode'] == 'full'
                assert hashlib.sha256(json.dumps(saved['physical'], sort_keys=True).encode()).hexdigest() == geom
                expected = reference_distances(case)
                assert saved['matrix'] == expected
                labels = saved['labels']
                for pair, path in saved['paths'].items():
                    a, b = pair.split('->')
                    assert path[0] == case['points'][a] and path[-1] == case['points'][b]
                    assert all(abs(u[0]-v[0])+abs(u[1]-v[1]) == 1 for u, v in zip(path, path[1:]))
                    assert sum(case['grid'][r][c] for r, c in path[1:]) == expected[labels.index(a)][labels.index(b)]
                distances[geom] = saved
            saved = distances[geom]
            assert saved['physical']['grid'] == case['grid'] and saved['physical']['points'] == case['points']
            key = digest, row['depth']
            if key not in refs:
                labels = saved['labels'][2:]
                values = [case['values'][t] for t in labels]
                ref = reference_game(saved['matrix'], values, row['depth'])
                if len(values) <= 4:
                    full, actions = enumerate_game(saved['matrix'], values, row['depth'])
                    assert full == ref['value'] and actions == ref['action_values']
                refs[key] = ref
            ref = refs[key]
            labels = saved['labels'][2:]
            assert row['expected_value'] == ref['value']
            assert row['optimal_actions'] == [labels[t] for t in ref['optimal_actions']]
            if row['status'] == 'ok':
                assert row['best_value'] == ref['value']
                assert row['best_target'] == labels[min(ref['optimal_actions'])]
                assert row['independently_verified']
            else:
                assert row['best_value'] is None and row['best_target'] is None
            rows.append(row)
    # 各主要随机条件必须覆盖完整20个固定种子，不能只保留完成得快的实例。
    main = [r for r in rows if r['experiment_id'] == 'Q4-03' and r.get('analysis_role') == 'ablation']
    assert len(main) == 300
    for depth in [4, 6, 8]:
        for alg in ['MM', 'AB', 'AB+Order', 'AB+TT', 'AB+Order+TT']:
            assert {r['seed'] for r in main if r['depth'] == depth and r['algorithm'] == alg} == set(range(10, 30))
    for axis, values in [('map_size', [7, 15, 30, 50]), ('treasure_count', [4, 6, 8, 10, 12]),
                         ('depth', [2, 4, 6, 8]), ('full_search_reference', [4, 6, 8])]:
        for value in values:
            for alg in ['MM', 'AB', 'AB+Order+TT']:
                assert {r['seed'] for r in rows if r['experiment_id'] == 'Q4-05' and r['axis'] == axis
                        and r['axis_value'] == value and r['algorithm'] == alg} == set(range(10, 30))
    # 开发集也保存输入并单独检查，开发输入不混入正式实验样本。
    for digest in development_hashes:
        raw = gzip.decompress((OUT/'instances'/f'{digest}.json.gz').read_bytes())
        assert hashlib.sha256(raw).hexdigest() == digest
    legacy_names = ['__init__.py', 'experiments.py', 'fixtures.py', 'generators.py', 'grid.py', 'reference.py', 'tree.py']
    old_hash = hashlib.sha256(b''.join((ROOT/'searchlab'/n).read_bytes() for n in legacy_names)).hexdigest()
    legacy_env = json.loads((OUT/'environment.json').read_text(encoding='utf-8'))
    assert measured_version(old_hash) == legacy_env['source_hash'], 'measured Q1-Q3 kernel was modified'
    # 完整行动轨迹及所有首步诊断也独立验证，不能只核对最终根值。
    continuations = json.loads((QOUT/'diagnostics/complete_continuations.json').read_text(encoding='utf-8'))
    for name, diagnostic in continuations.items():
        case, matrix = diagnostic['case'], diagnostic['matrix']
        labels = diagnostic['labels'][2:]
        ref = reference_game(matrix, [case['values'][t] for t in labels], 4)
        verify_trace(diagnostic['optimal_line'], matrix, case, ref['value'])
        for kind in ['fixed_greedy_first_line', 'fixed_shallow_first_line']:
            line = diagnostic[kind]
            first = labels.index(line[0]['target'])
            verify_trace(line, matrix, case, ref['action_values'][first], True)
    teacher = json.loads((QOUT/'diagnostics/teacher.json').read_text(encoding='utf-8'))
    for horizon, diagnostic in teacher['by_depth'].items():
        ref = reference_game(teacher['matrix'], [teacher['case']['values'][t] for t in teacher['labels'][2:]], int(horizon))
        assert diagnostic['action_values'] == {teacher['labels'][t+2]: v for t, v in ref['action_values'].items()}
        verify_trace(diagnostic['line'], teacher['matrix'], teacher['case'], ref['value'])
    sweeps = json.loads((QOUT/'diagnostics/value_sweeps.json').read_text(encoding='utf-8'))
    assert len(sweeps) == 124
    for diagnostic in sweeps:
        base = continuations[diagnostic['case']]
        case = {**base['case'], 'values': {**base['case']['values'], diagnostic['target']: diagnostic['value']}}
        ref = reference_game(base['matrix'], [case['values'][t] for t in base['labels'][2:]], diagnostic['depth'])
        assert diagnostic['action_values'] == {base['labels'][t+2]: v for t, v in ref['action_values'].items()}
        assert diagnostic['all_optimal_actions'] == [base['labels'][t+2] for t in ref['optimal_actions']]
        verify_trace(diagnostic['line'], base['matrix'], case, ref['value'])
    report = dict(status='passed', experiment_groups=5, run_rows=len(rows),
                  counts=dict(Counter(r['experiment_id'] for r in rows)),
                  statuses=dict(Counter(r['status'] for r in rows)),
                  unique_test_instances=len(cases), development_instances=len(development_hashes),
                  unique_test_geometries=len(distances), source_hash=current_hash,
                  independent_distance_and_action_values_verified=True,
                  canonical_ties_verified=True, exhaustive_small_cases_verified=True,
                  root_action_diagnostics_and_full_continuations_verified=True,
                  development_test_separation_verified=True, legacy_measured_kernel_unchanged=True)
    (QOUT/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    combined = []
    for path in sorted((OUT/'raw').glob('Q[1234]-*.jsonl')):
        combined.extend(json.loads(s) for s in path.read_text(encoding='utf-8').splitlines())
    fields = sorted(set().union(*(r.keys() for r in combined)))
    with (OUT/'results_all.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in combined:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    combined_summary = dict(experiment_groups=len({r['experiment_id'] for r in combined}), rows=len(combined),
                            counts=dict(Counter(r['experiment_id'] for r in combined)),
                            statuses=dict(Counter(r['status'] for r in combined)))
    (OUT/'run_summary_all.json').write_text(json.dumps(combined_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
