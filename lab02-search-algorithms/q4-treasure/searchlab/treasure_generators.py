# Q4 受控随机案例：局部种子、最大连通分量、配对几何和嵌套宝藏。
from collections import deque
import math

from .generators import local_random


def components(grid):
    unseen = {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v is not None}
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue, component = deque([start]), [start]
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                point = r+dr, c+dc
                if point in unseen:
                    unseen.remove(point)
                    queue.append(point)
                    component.append(point)
        result.append(sorted(component))
    return sorted(result, key=lambda part: (-len(part), part[0]))


def random_case(size, treasures, seed, family, value_range=(10, 30)):
    rng = local_random('Q4_'+family, seed, size=size, treasures=treasures, values=value_range)
    for attempt in range(1, 1001):
        grid = [[None if rng.random() < .15 else rng.choices([1, 3, 9], [.8, .15, .05])[0]
                 for _ in range(size)] for _ in range(size)]
        parts = components(grid)
        if not parts or len(parts[0]) < treasures+2:
            continue
        points = rng.sample(parts[0], treasures+2)
        labels = ['A', 'B']+[f'T{i+1}' for i in range(treasures)]
        for r, c in points:
            grid[r][c] = 1
        return dict(id=f'{family}_{size}_{treasures}_{seed}', grid=grid,
                    points={t: list(p) for t, p in zip(labels, points)},
                    values={t: rng.randint(*value_range) for t in labels[2:]},
                    generation=dict(version='q4_v1', seed=seed, family=family, attempts=attempt,
                                    obstacle_probability=.15, actual_density=sum(v is None for row in grid for v in row)/size**2))
    raise RuntimeError('generation_failed')


def subset_case(case, count):
    labels = ['A', 'B']+[f'T{i+1}' for i in range(count)]
    return {**case, 'id': case['id']+f'_prefix{count}',
            'points': {k: case['points'][k] for k in labels},
            'values': {k: case['values'][k] for k in labels[2:]}}


def scaled_cases(seed, sizes=(7, 15, 30, 50)):
    # 先生成粗地图和8个点，若任一尺寸无法保持互达，则整组重采样。
    rng = local_random('Q4_paired_geometry', seed)
    labels = ['A', 'B']+[f'T{i+1}' for i in range(6)]
    for attempt in range(1, 1001):
        coarse = [[None if rng.random() < .15 else 1 for _ in range(7)] for _ in range(7)]
        parts = components(coarse)
        if not parts or len(parts[0]) < 8:
            continue
        points = rng.sample(parts[0], 8)
        values = [rng.randint(10, 30) for _ in range(6)]
        result = []
        for size in sizes:
            source = lambda x: math.floor(6*x/(size-1)+.5)
            grid = [[coarse[source(r)][source(c)] for c in range(size)] for r in range(size)]
            available = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v is not None]
            chosen = []
            for cr, cc in points:
                nominal = math.floor(cr*(size-1)/6+.5), math.floor(cc*(size-1)/6+.5)
                candidates = [p for p in available if p not in chosen]
                if not candidates:
                    break
                chosen.append(min(candidates, key=lambda p: (abs(p[0]-nominal[0])+abs(p[1]-nominal[1]), p)))
            if len(chosen) != 8:
                break
            if not any(set(chosen) <= set(part) for part in components(grid)):
                break
            result.append(dict(id=f'scale_geometry_{seed}_{size}', grid=grid,
                               points={k: list(p) for k, p in zip(labels, chosen)},
                               values={k: math.floor(v*(size-1)/6+.5) for k, v in zip(labels[2:], values)},
                               generation=dict(version='q4_v1', family='paired_geometry', seed=seed,
                                               attempts=attempt, coarse_grid=coarse, coarse_points=points,
                                               coarse_values=values, value_scale=(size-1)/6)))
        if len(result) == len(sizes):
            return result
    raise RuntimeError('generation_failed: paired sizes')
