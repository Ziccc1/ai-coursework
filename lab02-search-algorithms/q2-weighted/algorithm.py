# 题目二接口：进入终点代价为0的加权A*。
import json
from pathlib import Path
from searchlab.grid import Grid, astar, astar_weighted, build_heuristic

def solve(data, *, heuristic=None, ordering='fixed', **options):
    grid = Grid(data['grid'], tuple(data['start']), tuple(data['goal']), 'weighted')
    h, prep = build_heuristic(grid, heuristic or 'corrected')
    result = astar(grid, heuristic=h, tie_break='larger_g' if ordering == 'fixed' else ordering)
    answer = result.to_dict()
    answer['heuristic_preprocess_ns'] = prep
    return answer

def sample():
    data = json.loads((Path(__file__).parent/'inputs/q2_teacher.json').read_text(encoding='utf-8-sig'))
    return {name: solve(data, heuristic=name) for name in ['zero', 'corrected', 'relaxed']}
