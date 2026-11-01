# Q4 独立参考：逐源 Dijkstra + 排列前缀回传 / 动态规划。
# 不调用正式状态转移、排序、缓存或 Alpha-beta 实现。
from functools import lru_cache
from itertools import permutations

from .reference import shortest_distances


def reference_distances(case):
    labels = ['A', 'B'] + sorted(case['values'], key=lambda t: int(t[1:]))
    result = []
    for label in labels:
        dist = shortest_distances(case['grid'], tuple(case['points'][label]))
        result.append([dist.get(tuple(case['points'][t]), -1) for t in labels])
    return result


def reference_game(matrix, values, depth, *, a=0, b=1, mask=None, player=0):
    # 返回增量值与每个首步值；使用独立状态递推，缓存容量不设实验限额。
    values = tuple(values)
    mask = (1 << len(values))-1 if mask is None else mask

    @lru_cache(None)
    def solve(pa, pb, available, who, horizon):
        if horizon == 0 or available == 0:
            return 0
        scores = []
        for t in range(len(values)):
            if available & (1 << t):
                if who == 0:
                    scores.append(values[t]-matrix[pa][t+2]+solve(t+2, pb, available & ~(1 << t), 1, horizon-1))
                else:
                    scores.append(matrix[pb][t+2]-values[t]+solve(pa, t+2, available & ~(1 << t), 0, horizon-1))
        return (max if who == 0 else min)(scores)

    actions = {}
    if depth and mask:
        for t in range(len(values)):
            if mask & (1 << t):
                if player == 0:
                    actions[t] = values[t]-matrix[a][t+2]+solve(t+2, b, mask & ~(1 << t), 1, depth-1)
                else:
                    actions[t] = matrix[b][t+2]-values[t]+solve(a, t+2, mask & ~(1 << t), 0, depth-1)
    best = (max if player == 0 else min)(actions.values()) if actions else 0
    result = dict(value=best, action_values=actions,
                  optimal_actions=[t for t, v in actions.items() if v == best],
                  reference_states=solve.cache_info().currsize)
    solve.cache_clear()
    return result


def enumerate_game(matrix, values, depth):
    # 小规模穷举所有终局排列，再由前缀逐层回传，独立于递归搜索。
    n = len(values)
    table = {}
    for sequence in permutations(range(n), depth):
        pa, pb, utility = 0, 1, 0
        for turn, target in enumerate(sequence):
            if turn % 2 == 0:
                utility += values[target]-matrix[pa][target+2]
                pa = target+2
            else:
                utility -= values[target]-matrix[pb][target+2]
                pb = target+2
        table[sequence] = utility
    for length in range(depth-1, -1, -1):
        for prefix in permutations(range(n), length):
            children = [table[prefix+(t,)] for t in range(n) if t not in prefix]
            table[prefix] = (max if length % 2 == 0 else min)(children)
    actions = {t: table[(t,)] for t in range(n)} if depth else {}
    return table[()], actions
