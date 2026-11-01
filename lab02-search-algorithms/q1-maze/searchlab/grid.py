# 四方向网格 A*：代价规则与搜索过程分离。
from dataclasses import asdict, dataclass, field
import heapq
import math
from time import perf_counter_ns

Cell = tuple[int, int]


@dataclass
class SearchResult:
    path: list[Cell] = field(default_factory=list)
    cost: int = -1
    expanded: int = 0
    generated: int = 0
    stale_pops: int = 0
    max_open: int = 0
    elapsed_ns: int = 0
    status: str = 'ok'
    expansion_order: list[Cell] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class Grid:
    def __init__(self, data, start, goal, problem='maze'):
        if problem not in ('maze', 'weighted', 'treasure'):
            raise ValueError('problem must be maze, weighted, or treasure')
        if not data or not data[0] or any(len(row) != len(data[0]) for row in data):
            raise ValueError('grid must be a nonempty rectangle')
        self.start, self.goal = tuple(start), tuple(goal)
        self.rows, self.cols = len(data), len(data[0])
        self.problem = problem
        self.costs = []
        for row in data:
            parsed = []
            for v in row:
                if v is None or v == 'X':
                    parsed.append(None)
                elif problem == 'maze':
                    if v not in ('.', 'S', 'E', 1):
                        raise ValueError(f'invalid maze cell: {v!r}')
                    parsed.append(1)
                elif isinstance(v, str) and (v in ('S', 'E', 'A', 'B') or v.startswith('T')):
                    parsed.append(1)
                else:
                    if isinstance(v, bool):
                        raise ValueError('boolean terrain costs are invalid')
                    try:
                        n = int(v)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f'invalid terrain cost: {v!r}') from exc
                    if n < 0 or isinstance(v, float) and n != v:
                        raise ValueError('terrain costs must be nonnegative integers')
                    parsed.append(n)
            self.costs.append(parsed)
        for point in [self.start, self.goal]:
            if len(point) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in point):
                raise ValueError('coordinates must be integer (row,col) pairs')
            if not (0 <= point[0] < self.rows and 0 <= point[1] < self.cols):
                raise ValueError('coordinate outside grid')
            if self.costs[point[0]][point[1]] is None:
                raise ValueError('start and goal must be traversable')
        if problem == 'weighted':
            self.costs[self.goal[0]][self.goal[1]] = 0
        self.c_min = min((v for r, row in enumerate(self.costs) for c, v in enumerate(row)
                          if v is not None and (r, c) != self.goal), default=0)

    def neighbors(self, node):
        r, c = node
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            a, b = r + dr, c + dc
            if 0 <= a < self.rows and 0 <= b < self.cols:
                cost = self.costs[a][b]
                if cost is not None:
                    yield (a, b), cost


def build_heuristic(grid, kind):
    # 返回启发函数与实际预处理时间。普通曼哈顿在 Q2 仅供反例诊断。
    begin = perf_counter_ns()
    goal = grid.goal
    if kind == 'relaxed':
        # 在放松图反向 Dijkstra：v->u 的反向边权是进入 v 的正向代价。
        costs = [[grid.c_min if v is None else v for v in row] for row in grid.costs]
        dist, queue = {goal: 0}, [(0, goal)]
        while queue:
            g, v = heapq.heappop(queue)
            if dist[v] != g:
                continue
            for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                u = v[0] + dr, v[1] + dc
                if 0 <= u[0] < grid.rows and 0 <= u[1] < grid.cols:
                    candidate = g + costs[v[0]][v[1]]
                    if candidate < dist.get(u, math.inf):
                        dist[u] = candidate
                        heapq.heappush(queue, (candidate, u))
        h = dist.__getitem__
    elif kind == 'zero':
        h = lambda n: 0
    elif kind == 'manhattan':
        h = lambda n: abs(n[0] - goal[0]) + abs(n[1] - goal[1])
    elif kind == 'euclidean':
        h = lambda n: math.hypot(n[0] - goal[0], n[1] - goal[1])
    elif kind == 'corrected':
        h = lambda n: max(abs(n[0] - goal[0]) + abs(n[1] - goal[1]) - 1, 0) * grid.c_min
    else:
        raise ValueError(f'unknown heuristic: {kind}')
    return h, perf_counter_ns() - begin


def astar(grid, start=None, goal=None, *, heuristic=None, tie_break='larger_g',
          problem='maze', record=False, max_expanded=10_000_000, seconds=60):
    # 以有效目标弹出作为终止条件；允许更优 g 重新入队，过滤过期项。
    if not isinstance(grid, Grid):
        grid = Grid(grid, start, goal, problem)
    if heuristic is None:
        heuristic = 'manhattan' if grid.problem == 'maze' else 'corrected'
    h = build_heuristic(grid, heuristic)[0] if isinstance(heuristic, str) else heuristic
    if tie_break not in ('larger_g', 'smaller_g', 'fifo'):
        raise ValueError('invalid tie_break')
    clock = perf_counter_ns()
    result = SearchResult(generated=1, max_open=1)
    seq = 0
    best, parent = {grid.start: 0}, {}

    def item(node, g, counter):
        tie = -g if tie_break == 'larger_g' else g if tie_break == 'smaller_g' else 0
        return g + h(node), tie, counter, g, node

    queue = [item(grid.start, 0, 0)]
    while queue:
        _, _, _, g, node = heapq.heappop(queue)
        if g != best[node]:
            result.stale_pops += 1
            continue
        if record:
            result.expansion_order.append(node)
        if node == grid.goal:
            result.cost = g
            route = [node]
            while route[-1] != grid.start:
                route.append(parent[route[-1]])
            result.path = route[::-1]
            break
        if result.expanded >= max_expanded:
            result.status = 'node_limit'
            break
        if result.expanded % 256 == 0 and perf_counter_ns() - clock > seconds * 1e9:
            result.status = 'timeout'
            break
        result.expanded += 1
        for nxt, cost in grid.neighbors(node):
            candidate = g + cost
            if candidate < best.get(nxt, math.inf):
                best[nxt], parent[nxt] = candidate, node
                seq += 1
                heapq.heappush(queue, item(nxt, candidate, seq))
                result.generated += 1
        result.max_open = max(result.max_open, len(queue))
    if not result.path and result.status == 'ok':
        result.status = 'unreachable'
    result.elapsed_ns = perf_counter_ns() - clock
    return result


def astar_maze(start, goal, grid):
    return astar(grid, start, goal, problem='maze').path


def astar_weighted(start, goal, grid):
    return astar(grid, start, goal, problem='weighted').cost
