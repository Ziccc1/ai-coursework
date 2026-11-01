# 本题运行入口；可从任意工作目录调用。
import argparse
import json
from pathlib import Path
import sys

from algorithm import solve, sample

ROOT = Path(__file__).resolve().parent
QUESTION = 4
GROUPS = [f'Q{QUESTION}-{i:02d}' for i in range(1, 6 if QUESTION == 4 else 5)]


def main():
    parser = argparse.ArgumentParser(description='题目{}：样例、求解、实验、绘图与核验'.format(QUESTION))
    sub = parser.add_subparsers(dest='command')
    sub.add_parser('samples', help='显示给定样例结果')
    task = sub.add_parser('solve', help='求解本题JSON输入；省略--input时使用给定样例')
    task.add_argument('--input', type=Path)
    task.add_argument('--depth', type=int)
    task.add_argument('--algorithm', choices=['MM', 'AB'], default='AB')
    task.add_argument('--heuristic')
    task.add_argument('--ordering', default='fixed')
    task.add_argument('--cache', action='store_true')
    ex = sub.add_parser('experiments', help='重跑本题实验，保存到outputs_reproduced')
    ex.add_argument('--ids', default='all')
    ex.add_argument('--resume', action='store_true')
    ex.add_argument('--timing', choices=['full', 'single'], default='full')
    sub.add_parser('plots', help='使用随附正式结果生成本题PNG')
    sub.add_parser('check', help='独立核验随附正式结果')
    args = parser.parse_args()
    if args.command in (None, 'samples'):
        answer = sample()
        print(json.dumps(answer, ensure_ascii=False, indent=2))
    elif args.command == 'solve':
        source = args.input or ROOT/'inputs'/f'q{QUESTION}_teacher.json'
        data = json.loads(source.read_text(encoding='utf-8-sig'))
        print(json.dumps(solve(data, depth=args.depth, algorithm=args.algorithm,
                               heuristic=args.heuristic, ordering=args.ordering,
                               cache=args.cache), ensure_ascii=False, indent=2))
    elif args.command == 'experiments':
        selected = GROUPS if args.ids == 'all' else args.ids.split(',')
        if any(i not in GROUPS for i in selected):
            parser.error('只能指定本题实验组：'+','.join(GROUPS))
        if QUESTION == 4:
            from searchlab import treasure_experiments as ex
            ex.OUT = ROOT/'outputs_reproduced'
            ex.QOUT = ex.OUT/'q4'
        else:
            from searchlab import experiments as ex
            ex.OUT = ROOT/'outputs_reproduced'
            ex.IDS = GROUPS
        ex.run(','.join(selected), resume=args.resume, timing=args.timing)
        print('本次重跑结果：'+str(ex.OUT))
    elif args.command == 'plots':
        if QUESTION == 4:
            from scripts.plot_treasure import main as plot
            plot()
        else:
            import csv
            from scripts import plot_results as plot
            plot.setup()
            plot.MANIFEST.clear()
            getattr(plot, f'q{QUESTION}_figures')(plot.read_rows())
            with (plot.FIG/'figure_manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id','path','claim','source','script'])
                writer.writeheader()
                writer.writerows(plot.MANIFEST)
            print(f'Created {len(plot.MANIFEST)} PNG figures')
    elif args.command == 'check':
        if QUESTION == 4:
            from scripts.validate_treasure import main as check
            check()
        else:
            from scripts import validate_outputs as check
            check.EXPECTED = {k:v for k,v in check.EXPECTED.items() if k.startswith(f'Q{QUESTION}-')}
            check.main()


if __name__ == '__main__':
    main()
