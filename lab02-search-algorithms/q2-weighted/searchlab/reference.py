# 独立正确性参考；仅供验证，不用于替代题目要求的算法。
from collections import deque
import heapq


def shortest_distances(costs, source, reverse=False):
    dist, queue = {source: 0}, [(0, source)]
    while queue:
        g, u = heapq.heappop(queue)
        if dist[u] != g:
            continue
        for dr, dc in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            v = u[0] + dr, u[1] + dc
            if 0 <= v[0] < len(costs) and 0 <= v[1] < len(costs[0]) and costs[v[0]][v[1]] is not None:
                ng = g + (costs[u[0]][u[1]] if reverse else costs[v[0]][v[1]])
                if ng < dist.get(v, float('inf')):
                    dist[v] = ng
                    heapq.heappush(queue, (ng, v))
    return dist


def bfs_route(costs, source, target):
    parent, queue = {source: None}, deque([source])
    while queue:
        u = queue.popleft()
        if u == target:
            result = []
            while u is not None:
                result.append(u)
                u = parent[u]
            return result[::-1]
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            v = u[0] + dr, u[1] + dc
            if 0 <= v[0] < len(costs) and 0 <= v[1] < len(costs[0]) and costs[v[0]][v[1]] is not None and v not in parent:
                parent[v] = u
                queue.append(v)
    return []


def check_route(grid, result, expected):
    if result.status not in ('ok', 'unreachable'):
        return
    assert result.cost == expected, (result.cost, expected)
    if expected < 0:
        assert not result.path
        return
    assert result.path[0] == grid.start and result.path[-1] == grid.goal
    assert sum(grid.costs[r][c] for r, c in result.path[1:]) == result.cost
    for u, v in zip(result.path, result.path[1:]):
        assert abs(u[0] - v[0]) + abs(u[1] - v[1]) == 1
        assert grid.costs[v[0]][v[1]] is not None


def reference_tree(root):
    # 对可达无环图独立进行后序求值，并检查是否存在环。
    stack, values, active = [(root, True, False)], {}, set()
    while stack:
        node, maximizing, ready = stack.pop()
        key = id(node), maximizing
        if key in values:
            continue
        if ready:
            active.discard(key)
            children = [values[id(c), not maximizing] for c in node.children]
            values[key] = (max(children) if maximizing else min(children)) if children else node.value
        else:
            if key in active:
                raise ValueError('cycle')
            active.add(key)
            stack.append((node, maximizing, True))
            stack.extend((c, not maximizing, False) for c in reversed(node.children))
    return values[id(root), True]
