# 从Q4正式原始数据生成可核查的结果摘要与完整行动诊断。
import gzip
import json
from pathlib import Path

import pandas as pd

from searchlab.treasure import TreasureProblem, search_game

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
QOUT = OUT/'q4'


def read_rows():
    rows = []
    for file in sorted((OUT/'raw').glob('Q4-*.jsonl')):
        rows.extend(json.loads(s) for s in file.read_text(encoding='utf-8').splitlines())
    return pd.DataFrame(rows)


def continuation(problem, first):
    state, line = problem.initial_state(), []
    for step in range(problem.count):
        target = first if step == 0 else search_game(problem, problem.count-step, state=state, cache=True).best_target
        index = problem.targets.index(target)
        origin = state.a_pos if state.current_player == 0 else state.b_pos
        nxt = problem.apply(state, index)
        line.append(dict(turn=step+1, player='A' if state.current_player == 0 else 'B',
                         origin=problem.labels[origin], target=target,
                         distance=problem.distances[origin][index+2], value=problem.values[index],
                         gain=problem.gain(origin, index), score_a=nxt.score_a, score_b=nxt.score_b,
                         utility=nxt.utility))
        state = nxt
    return line


def switch_ranges(sweep):
    ranges = []
    for row in sorted(sweep, key=lambda r: r['value']):
        optimal = row['all_optimal_actions']
        if ranges and ranges[-1]['actions'] == optimal and row['value'] == ranges[-1]['stop']+1:
            ranges[-1]['stop'] = row['value']
        else:
            ranges.append(dict(start=row['value'], stop=row['value'], actions=optimal))
    return ranges


def paired_summary(frame, baseline, methods, metric):
    table = frame.pivot(index='seed', columns='algorithm', values=metric)
    return {m: float((table[m]/table[baseline]).median()) for m in methods}


def main():
    frame = read_rows()
    if set(frame.experiment_id) != {f'Q4-{i:02d}' for i in range(1, 6)}:
        raise ValueError('all five Q4 experiment groups are required')
    diagnostics = QOUT/'diagnostics'
    paths = {}
    scenarios = []
    for case_id in ['teacher', 'C1', 'C2', 'C3', 'C4']:
        group = frame[(frame.experiment_id == ('Q4-01' if case_id == 'teacher' else 'Q4-02')) & (frame.instance_name == case_id)]
        row = group.iloc[0]
        payload = json.loads(gzip.decompress((OUT/'instances'/(row.instance_sha256+'.json.gz')).read_bytes()))
        distance = json.loads((QOUT/'distances'/(row.geometry_sha256+'.json')).read_text(encoding='utf-8'))
        problem = TreasureProblem(payload['case'], distance['matrix'])
        full = search_game(problem, 4, cache=True)
        greedy = max(range(4), key=lambda i: (problem.gain(0, i), -i))
        shallow = search_game(problem, 2)
        optimal_line = continuation(problem, full.best_target)
        greedy_line = continuation(problem, problem.targets[greedy])
        shallow_line = continuation(problem, shallow.best_target)
        assert optimal_line[-1]['utility'] == full.best_value
        scenarios.append(dict(case=case_id, greedy=problem.targets[greedy], greedy_full_value=greedy_line[-1]['utility'],
                              shallow=shallow.best_target, shallow_full_value=shallow_line[-1]['utility'],
                              optimal=full.best_target, full_value=full.best_value,
                              greedy_regret=full.best_value-greedy_line[-1]['utility'],
                              shallow_regret=full.best_value-shallow_line[-1]['utility']))
        paths[case_id] = dict(case=payload['case'], labels=distance['labels'], matrix=distance['matrix'],
                              paths=distance['paths'], optimal_line=optimal_line,
                              fixed_greedy_first_line=greedy_line, fixed_shallow_first_line=shallow_line)
    (diagnostics/'complete_continuations.json').write_text(json.dumps(paths, ensure_ascii=False, indent=2), encoding='utf-8')
    selection = json.loads((QOUT/'ordering_selection.json').read_text(encoding='utf-8'))
    ablation = frame[(frame.experiment_id == 'Q4-03') & (frame.analysis_role == 'ablation')]
    methods = ['MM', 'AB', 'AB+Order', 'AB+TT', 'AB+Order+TT']
    ablation_rows, paired = [], {}
    for depth in [4, 6, 8]:
        subset = ablation[ablation.depth == depth]
        common = subset.pivot(index='seed', columns='algorithm', values='status').eq('ok').all(axis=1)
        complete = subset[subset.seed.isin(common[common].index)]
        paired[str(depth)] = dict(common_completed=int(common.sum()),
                                  calls_ratio_to_MM=paired_summary(complete, 'MM', methods, 'calls'),
                                  time_ratio_to_MM=paired_summary(complete, 'MM', methods, 'elapsed_ns'))
        for alg in methods:
            sub = subset[subset.algorithm == alg]
            ablation_rows.append(dict(depth=depth, algorithm=alg, calls=float(sub.calls.median()),
                                      elapsed_ms=float(sub.elapsed_ns.median()/1e6),
                                      tt_hit_rate=float((sub.tt_hits/sub.tt_probes.replace(0, float('nan'))).median()) if sub.tt_probes.max() else None,
                                      completion_rate=float(sub.status.eq('ok').mean())))
    ordering = frame[(frame.experiment_id == 'Q4-03') & (frame.depth == 6) & ~frame.cache & frame.pruning]
    ordering_rows = []
    for order in ['fixed', 'immediate', 'competition']:
        if order == 'immediate':
            sub = ordering[(ordering.ordering == order) | ((ordering.ordering == 'competition') & (ordering.eta == 0))]
        else:
            sub = ordering[(ordering.ordering == order) & ((ordering.eta > 0) if order == 'competition' else True)]
        ordering_rows.append(dict(ordering=order, instances=len(sub), calls=float(sub.calls.median()),
                                  elapsed_ms=float(sub.elapsed_ns.median()/1e6)))
    sweeps = json.loads((diagnostics/'value_sweeps.json').read_text(encoding='utf-8'))
    sensitivity = {}
    for case in ['teacher', 'C3']:
        sensitivity[case] = {str(h): switch_ranges([r for r in sweeps if r['case'] == case and r['depth'] == h]) for h in [2, 4]}
    scale_rows = []
    scale = frame[frame.experiment_id == 'Q4-05']
    for (axis, value, alg), sub in scale.groupby(['axis', 'axis_value', 'algorithm']):
        ok = sub[sub.status == 'ok']
        scale_rows.append(dict(axis=axis, axis_value=int(value), algorithm=alg,
                               complete=len(ok), total=len(sub), calls=float(ok.calls.median()),
                               search_ms=float(ok.elapsed_ns.median()/1e6),
                               preprocess_ms=float(ok.preprocess_ns.median()/1e6),
                               end_to_end_ms=float(ok.end_to_end_ns.median()/1e6),
                               normalized_utility=float((ok.best_value/(ok['size']-1)).median())))
    summary = dict(rows=len(frame), selected_eta=selection['selected_eta'], scenarios=scenarios,
                   ablation=ablation_rows, paired_ablation=paired, ordering=ordering_rows,
                   sensitivity_integer_ranges=sensitivity, scaling=scale_rows,
                   caveats=['artificial cases are explanatory curated examples',
                            'decision regret uses a common full-depth action-value reference',
                            'paired ratios are calculated per identical instance before aggregation',
                            'normalised map generation does not make different grids exactly equivalent',
                            'optimal-action ties are retained in the value sweeps'])
    (QOUT/'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# 第四题实际实验结果与分析', '',
             f'共完成 {len(frame)} 条配置结果，五组实验均保存完整原始记录。开发集与测试集分离，竞争排序参数固定为 eta={selection["selected_eta"]:g}。', '',
             '## 基础任务：原题与深度', '',
             'A*计算全部30个非对角有序点对，特殊格进入代价为1。Minimax与Alpha-beta均得到 H=2：T2、-1；H=4：T3、3。', '',
             '原题完整收益差为3，而固定浅层首步T2后双方最优应对的完整值为1，故首步损失为2；不能将不同深度的3和-1直接相减。', '',
             '## 人工场景与策略分析', '',
             '| 案例 | 贪心首步 | 固定该首步的完整值 | 浅层首步 | 完整最优首步 | 最优值 | 贪心首步损失 |',
             '|---|---|---:|---|---|---:|---:|']
    for row in scenarios:
        lines.append(f'| {row["case"]} | {row["greedy"]} | {row["greedy_full_value"]} | {row["shallow"]} | {row["optimal"]} | {row["full_value"]} | {row["greedy_regret"]} |')
    lines += ['', '上述人工案例经过定向设计，用于解释位置、竞争与绕行机制，不用于估计随机场景平均收益。完整延续序列见 diagnostics/complete_continuations.json。', '',
              'C1中，贪心先取T1的完整值为2，H=2先取T4的完整值反而为-3，而完整搜索先取T2的值为6。这表明在固定截断评价下，多看一步并不保证首步质量逐层单调改善。C3中H=2已能识别双方争抢T2的影响，选中完整最优首步，说明也不能将所有浅层选择都视为错误。', '',
              'C1与C2体现后续位置和移动成本，C3侧重双方竞争与宝藏分配，C4区分几何距离、最少步数和最小进入代价。具体路径和逐步收益均随诊断文件保存，解释时不能把整个收益改善都归因于某一方的路程节省。', '',
              '## 优化消融与计算开销', '',
              '| 深度 | 版本 | 访问次数中位数 | 搜索时间中位数/ms | 完成率 |', '|---:|---|---:|---:|---:|']
    for row in ablation_rows:
        lines.append(f'| {row["depth"]} | {row["algorithm"]} | {row["calls"]:g} | {row["elapsed_ms"]:.3f} | {row["completion_rate"]:.0%} |')
    lines += ['', '访问数包含根、叶和缓存直接返回；排序、置换表维护及规范首步所需的重搜均计入时间。距离矩阵、独立参考和完整首步诊断不计入博弈搜索时间。', '',
              '本批H=4结果中，带缓存版本的访问数中位数未下降，时间却略有增加。此时可复用的重复局面主要出现在截断叶层，叶层直接评价而不写入置换表，缓存维护难以省去后续递归。H增加到6、8后，重复局面具有未搜索的后继，缓存才有机会复用整段后续计算。排序与缓存合用在深层效果更好，但不保证在浅层也最快。', '',
              '| H=6排序方式 | 测试地图数 | 访问数中位数 | 时间中位数/ms |', '|---|---:|---:|---:|']
    for row in ordering_rows:
        lines.append(f'| {row["ordering"]} | {row["instances"]} | {row["calls"]:g} | {row["elapsed_ms"]:.3f} |')
    lines += ['', '竞争排序的访问数中位数小于即时排序，但时间中位数略高，说明额外计算对手收益也有成本。eta=1按预先规定的开发集节点指标选定，不代表它在测试集时间上必然最优。不能直接将排序与缓存各自的加速倍数相乘；配对时间和节点比值保存在analysis.json中。', '',
              '## 价值敏感性', '', '以下仅表示扫描整数范围内的最优行动集合，不宣称连续实数域的精确临界点。', '']
    for case, horizons in sensitivity.items():
        for horizon, ranges in horizons.items():
            text = '；'.join(f'{r["start"]}—{r["stop"]}：{",".join(r["actions"])}' for r in ranges)
            lines.append(f'- {case}，H={horizon}：{text}。')
    lines += ['', '原题完整搜索在T3价值为6时出现T2、T3并列，价值7时选择T3，而H=2到价值8时才改变首步。价值扫描不仅说明选择是否改变，也展示双方分配如何影响收益：在部分区间内，某宝藏由B取得，提高其价值反而会降低A-B，而非始终有利于A。', '',
              '## 规模与瓶颈', '',
              '| 变化因素 | 取值 | 版本 | 访问数中位数 | 预计算/ms | 搜索/ms | 总成本/ms | 完成数 |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in scale_rows:
        lines.append(f'| {row["axis"]} | {row["axis_value"]} | {row["algorithm"]} | {row["calls"]:g} | {row["preprocess_ms"]:.3f} | {row["search_ms"]:.3f} | {row["end_to_end_ms"]:.3f} | {row["complete"]}/{row["total"]} |')
    mapped = {(r['axis'], r['axis_value'], r['algorithm']): r for r in scale_rows}
    small, large = mapped['map_size', 7, 'AB+Order+TT'], mapped['map_size', 50, 'AB+Order+TT']
    plain, optimized = mapped['depth', 8, 'MM'], mapped['depth', 8, 'AB+Order+TT']
    lines += ['', f'配对地图规模由7增至50时，距离预计算时间中位数由{small["preprocess_ms"]:.3f} ms增至{large["preprocess_ms"]:.3f} ms，而固定6个宝藏、H=6的优化博弈搜索约为{small["search_ms"]:.3f}—{large["search_ms"]:.3f} ms，说明此组主要成本在网格距离计算。固定10个宝藏、H=8时，Minimax与组合优化的搜索中位数分别为{plain["search_ms"]:.3f} ms和{optimized["search_ms"]:.3f} ms，则体现博弈搜索分支增长带来的成本。两类瓶颈需要分开评价。']
    lines += ['', '地图规模、宝藏数量和搜索深度分别控制。地图规模组使用配对归一化粗几何并缩放价值，但离散化会改变实际距离，不能将各尺寸视为完全等价的同一博弈。', '',
              '每个实例先分别取得预计算和搜索的重复计时中位数，再相加为该实例的总成本；随后对20个实例汇总。因此，表中两列跨实例中位数不必恰好相加等于总成本中位数。这是成本构成统计，不是对两阶段联合计时。内存使用tracemalloc单独测量，只统计搜索期间的Python分配，相关运行不进入主时间比较。', '',
              '所有完成结果均由独立动态规划参考核验；四宝藏案例还通过全排列前缀回传校验。完整配置和计时批次保存在outputs/raw/Q4-*.jsonl。']
    (QOUT/'实验结果分析.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Q4 analysis: {len(frame)} rows, {len(scale_rows)} scale summaries')


if __name__ == '__main__':
    main()
