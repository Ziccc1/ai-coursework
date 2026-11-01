# 题目四接口：A*距离矩阵与有限深度宝藏博弈。
import json
from pathlib import Path
from searchlab.treasure import GameState, TreasureProblem, build_distances, search_game

def solve(data, *, depth=None, algorithm='AB', ordering='fixed', cache=False, **options):
    distances = build_distances(data)
    result = search_game(TreasureProblem(data, distances.matrix),
                         depth if depth is not None else data.get('depth', 4),
                         pruning=algorithm == 'AB', ordering=ordering, cache=cache, eta=1)
    return dict(labels=distances.labels, distance_matrix=distances.matrix, **result.to_dict())

def sample():
    data = json.loads((Path(__file__).parent/'inputs/q4_teacher.json').read_text(encoding='utf-8-sig'))
    return {f'{name}_H{depth}': solve(data, depth=depth, algorithm=name) for depth in [2,4] for name in ['MM','AB']}
