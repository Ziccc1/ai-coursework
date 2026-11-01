# 题目一接口：单位移动代价的迷宫A*。
import json
from pathlib import Path
from searchlab.grid import Grid, astar, astar_maze, build_heuristic

def solve(data, *, heuristic=None, ordering='fixed', **options):
    grid = Grid(data['grid'], tuple(data['start']), tuple(data['goal']), 'maze')
    h, prep = build_heuristic(grid, heuristic or 'manhattan')
    result = astar(grid, heuristic=h, tie_break='larger_g' if ordering == 'fixed' else ordering)
    return result.to_dict()

def sample():
    data = json.loads((Path(__file__).parent/'inputs/q1_teacher.json').read_text(encoding='utf-8-sig'))
    return {name: solve(data, heuristic=name) for name in ['zero', 'euclidean', 'manhattan']}
