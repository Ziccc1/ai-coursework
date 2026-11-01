# 逐题重新计算全部配置，核对答案与确定性搜索统计。
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent
FOLDERS = ['q1-maze', 'q2-weighted', 'q3-alpha-beta', 'q4-treasure']
EXPECTED = [564, 464, 1623, 1548]
LOCAL = ROOT/'verification/local'
COMMENT_REVISION = json.loads((ROOT/'verification/comment_revision.json').read_text(encoding='utf-8'))


def read_rows(folder):
    rows = []
    for path in sorted(folder.glob('Q*-*.jsonl')):
        rows.extend(json.loads(s) for s in path.read_text(encoding='utf-8').splitlines())
    indexed = {r['run_key']: r for r in rows}
    if len(indexed) != len(rows):
        raise AssertionError('存在重复run_key：'+str(folder))
    return indexed


def stable_fields(row):
    # 仅移除明确与计时流程相关的字段，保留全部答案、输入、参数和搜索计数。
    result = {k:v for k,v in row.items() if not k.endswith('_ns')
              and k not in ('timing_mode', 'batch_size')}
    # 注释修订有独立的前后校验记录；比较时将已确认的源码版本对应起来。
    if 'source_hash' in result:
        result['source_hash'] = COMMENT_REVISION['source_hash_aliases'].get(result['source_hash'], result['source_hash'])
    return result


def call(folder, args, log):
    command = [sys.executable, '-X', 'utf8', *args]
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\n$ '+' '.join(command)+'\n')
        stream.flush()
        done = subprocess.run(command, cwd=folder, stdout=stream, stderr=subprocess.STDOUT)
    if done.returncode:
        raise RuntimeError('命令失败，详见 '+str(log))


def teacher_result(folder):
    result = subprocess.run([sys.executable, '-X', 'utf8', 'main.py'], cwd=folder,
                            capture_output=True, text=True, encoding='utf-8', check=True)
    return json.loads(result.stdout)


def verify_question(q):
    LOCAL.mkdir(parents=True, exist_ok=True)
    folder = ROOT/FOLDERS[q-1]
    log = LOCAL/f'q{q}.log'
    log.write_text('', encoding='utf-8')
    begin = perf_counter()
    print(f'Q{q} START：核验样例、测试、正式记录，并重新求解全部配置', flush=True)
    baseline_manifest = json.loads((ROOT/'verification/baseline_manifest.json').read_text(encoding='utf-8'))
    checked_files = 0
    for relative, digest in baseline_manifest.items():
        if relative.startswith(FOLDERS[q-1]+'/'):
            actual = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
            if actual != digest:
                revision = COMMENT_REVISION['files'].get(relative, {})
                assert revision.get('before_sha256') == digest and revision.get('after_sha256') == actual, relative
            checked_files += 1
    sample = teacher_result(folder)
    if q == 1:
        assert all(v['cost'] == 7 and len(v['path']) == 8 for v in sample.values())
        assert [sample[h]['expanded'] for h in ['zero','euclidean','manhattan']] == [15,10,7]
        examples = dict(steps=7, coordinates=8, expanded=[15,10,7])
    elif q == 2:
        assert all(v['cost'] == 7 for v in sample.values())
        assert [sample[h]['expanded'] for h in ['zero','corrected','relaxed']] == [14,10,8]
        examples = dict(cost=7, expanded=[14,10,8])
    elif q == 3:
        assert sample['MM']['value'] == sample['AB']['value'] == 5
        assert [sample[a]['calls'] for a in ['MM','AB']] == [15,14]
        assert [sample[a]['leaf_evals'] for a in ['MM','AB']] == [8,7]
        examples = dict(value=5, calls=[15,14], leaf_evals=[8,7])
    else:
        for algorithm in ['MM','AB']:
            assert (sample[f'{algorithm}_H2']['best_target'], sample[f'{algorithm}_H2']['best_value']) == ('T2',-1)
            assert (sample[f'{algorithm}_H4']['best_target'], sample[f'{algorithm}_H4']['best_value']) == ('T3',3)
        examples = dict(H2=['T2',-1], H4=['T3',3], distance_matrix=sample['AB_H4']['distance_matrix'])
    call(folder, ['-m','unittest','discover','-s','tests','-v'], log)
    call(folder, ['main.py','check'], log)
    # 由实际分题入口重跑；正式输出目录不参与写入。
    call(folder, ['main.py','experiments','--ids','all','--timing','single'], log)
    old = read_rows(folder/'outputs/raw')
    new = read_rows(folder/'outputs_reproduced/raw')
    assert len(old) == EXPECTED[q-1], (q,len(old))
    problems = []
    if old.keys() != new.keys():
        problems.append(dict(kind='configuration_coverage', missing=sorted(old.keys()-new.keys()),
                             extra=sorted(new.keys()-old.keys())))
    fields = set()
    for key in sorted(old.keys() & new.keys()):
        before, after = stable_fields(old[key]), stable_fields(new[key])
        fields.update(before)
        if before != after:
            differences = {k:dict(before=before.get(k), after=after.get(k))
                           for k in sorted(before.keys() | after.keys()) if before.get(k) != after.get(k)}
            problems.append(dict(run_key=key, differences=differences))
    report = dict(question=q, folder=FOLDERS[q-1], status='passed' if not problems else 'failed',
                  baseline_rows=len(old), rerun_rows=len(new), matched_rows=len(old)-len(problems),
                  baseline_files_sha256_verified=checked_files, teacher=examples,
                  correctness_tests='passed', independent_saved_answer_checks='passed',
                  compared_fields=sorted(fields), excluded_fields='所有以_ns结尾的计时字段、timing_mode、batch_size',
                  rerun_mode='single: functional equality only; not a performance measurement',
                  verification_seconds=round(perf_counter()-begin,3), mismatches=problems)
    (LOCAL/f'q{q}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if problems:
        raise AssertionError(f'Q{q}存在{len(problems)}项差异，详见verification/local/q{q}.json')
    print(f'Q{q} PASS：{len(old)}条配置的答案与确定性统计逐条一致', flush=True)
    return report


def collect():
    questions = [json.loads((LOCAL/f'q{q}.json').read_text(encoding='utf-8')) for q in range(1,5)]
    assert all(q['status'] == 'passed' and not q['mismatches'] for q in questions)
    report = dict(status='passed', experiment_groups=17, matched_rows=sum(q['matched_rows'] for q in questions),
                  scope='全部17组重新求解，与原正式记录逐条比较答案、参数、输入哈希和确定性统计',
                  performance_rebenchmarked=False, questions=questions)
    (ROOT/'verification/equivalence_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 分题版本的一致性核验', '',
             '四个分题入口已重新计算全部17组实验。与原统一工程正式记录逐条比较后，4199条配置全部一致，无答案或确定性搜索统计差异。', '',
             '| 题目 | 重新求解的配置数 | 答案及搜索统计 |', '|---|---:|---|']
    for q in questions:
        lines.append(f"| Q{q['question']} | {q['matched_rows']} | 一致 |")
    lines += ['', '## 样例结果', '',
              '- 题目一：最短7步，8个坐标；零启发、欧氏距离、曼哈顿距离分别扩展15、10、7个节点。',
              '- 题目二：三种安全启发函数均得最小代价7，分别扩展14、10、8个节点。',
              '- 题目三：根值均为5；Minimax调用15次、评估8个叶节点，Alpha-Beta调用14次、评估7个叶节点。',
              '- 题目四：H=2均为T2、-1；H=4均为T3、3；距离矩阵一致。', '',
              '## 核验方法', '',
              '先用baseline_manifest.json检查各题算法与正式记录的SHA-256；注释修订通过comment_revision.json中的前后校验值对应。再执行各题正确性测试和正式结果的独立参考核验，最后通过main.py experiments重跑全部实验组。按run_key配对，比较除明确计时字段以外的所有字段，已确认的注释源码版本按修订记录对应。', '',
              '核对内容包括路径坐标、路径代价、最优首步、博弈值、距离矩阵相关统计、输入哈希、算法参数、节点扩展数、递归次数、叶节点评估次数、剪枝次数及缓存统计。部分指标仅适用于相应题目，精确字段列表保存在equivalence_report.json。', '',
              '本次采用single模式重新求解全部配置，用于验证整理目录后算法行为是否保持一致。耗时、预处理时间、计时批次大小和计时模式不要求一致，不能用本次运行替换原正式性能表。正式多批次计时数据仍保存在各题outputs/。', '',
              '复现命令：在项目根目录执行python verify.py。重跑输出与过程日志位于Git忽略的outputs_reproduced/和verification/local/，最终核验报告保存在verification/。']
    (ROOT/'verification/RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('ALL PASS：4199/4199；报告：verification/RESULTS.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description='逐题核验样例、正确性测试与实验结果')
    parser.add_argument('--question', type=int, choices=range(1,5), help='仅核验某题')
    parser.add_argument('--collect-only', action='store_true', help='汇总本次已有的四题核验记录')
    args = parser.parse_args()
    if args.question:
        verify_question(args.question)
    elif args.collect_only:
        collect()
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for finished in as_completed([pool.submit(verify_question,q) for q in range(1,5)]):
                finished.result()
        collect()


if __name__ == '__main__':
    main()
