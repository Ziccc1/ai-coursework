# 第四题：A* 距离矩阵、宝藏状态与 Minimax/Alpha-beta。
# 缓存后续增量收益 F，而非带历史累计分数的绝对 U。
# 边的增量为 delta 时，递归窗口必须平移为 (alpha-delta,beta-delta)。
from dataclasses import asdict, dataclass, field
import math
from time import perf_counter_ns

from .grid import Grid, astar


@dataclass(frozen=True)
class GameState:
    a_pos: int = 0
    b_pos: int = 1
    remaining_mask: int = 0
    score_a: int = 0
    score_b: int = 0
    current_player: int = 0  # 0表示玩家A的最大化回合，1表示玩家B的最小化回合。

    @property
    def utility(self):
        return self.score_a - self.score_b


@dataclass
class DistanceResult:
    labels: list
    matrix: list
    paths: dict
    searches: int = 0
    expanded: int = 0
    generated: int = 0
    elapsed_ns: int = 0
    pair_stats: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class TreasureProblem:
    def __init__(self, case, distances):
        self.case = case
        self.targets = sorted(case['values'], key=lambda t: int(t[1:]))
        self.labels = ['A', 'B'] + self.targets
        if set(case['points']) != set(self.labels):
            raise ValueError('points must contain A, B and every treasure exactly once')
        self.points = [tuple(case['points'][k]) for k in self.labels]
        if len(set(self.points)) != len(self.points):
            raise ValueError('special positions must be distinct')
        grid = Grid(case['grid'], self.points[0], self.points[1], 'treasure')
        if any(v is not None and v < 1 for row in grid.costs for v in row):
            raise ValueError('Q4 terrain costs must be positive integers')
        for point in self.points:
            if len(point) != 2 or any(type(x) is not int for x in point):
                raise ValueError('invalid special coordinate')
            r, c = point
            if not (0 <= r < grid.rows and 0 <= c < grid.cols) or grid.costs[r][c] != 1:
                raise ValueError('all special cells must have entering cost 1')
        self.values = tuple(case['values'][t] for t in self.targets)
        if not self.values or any(type(v) is not int for v in self.values):
            raise ValueError('treasures need integer values')
        n = len(self.labels)
        if len(distances) != n or any(len(row) != n for row in distances):
            raise ValueError('distance matrix has wrong dimensions')
        if any(type(v) is not int or v < 0 for row in distances for v in row):
            raise ValueError('all special positions must be mutually reachable')
        if any(distances[i][i] != 0 for i in range(n)):
            raise ValueError('distance diagonal must be zero')
        self.distances = tuple(tuple(row) for row in distances)
        self.count = len(self.targets)

    def initial_state(self):
        return GameState(remaining_mask=(1 << self.count)-1)

    def actions(self, state):
        return [i for i in range(self.count) if state.remaining_mask & (1 << i)]

    def gain(self, pos, target):
        return self.values[target] - self.distances[pos][target+2]

    def apply(self, state, target):
        if not state.remaining_mask & (1 << target):
            raise ValueError('treasure already taken')
        mask = state.remaining_mask ^ (1 << target)
        if state.current_player == 0:
            return GameState(target+2, state.b_pos, mask,
                             state.score_a+self.gain(state.a_pos, target), state.score_b, 1)
        return GameState(state.a_pos, target+2, mask, state.score_a,
                         state.score_b+self.gain(state.b_pos, target), 0)


def build_distances(case):
    # 完整有序点对用 A* 求解，特殊终点仍付进入代价 1。
    labels = ['A', 'B'] + sorted(case['values'], key=lambda t: int(t[1:]))
    points = [tuple(case['points'][k]) for k in labels]
    # 先验证输入，不能把不可达点对或算法中止静默记作合法距离。
    TreasureProblem(case, [[0]*len(labels) for _ in labels])
    result = DistanceResult(labels, [], {})
    begin = perf_counter_ns()
    for i, source in enumerate(points):
        row = []
        for j, destination in enumerate(points):
            if i == j:
                row.append(0)
                result.paths[f'{labels[i]}->{labels[j]}'] = [list(source)]
                continue
            grid = Grid(case['grid'], source, destination, 'treasure')
            h = lambda p, goal=destination, cmin=grid.c_min: cmin*(abs(p[0]-goal[0])+abs(p[1]-goal[1]))
            found = astar(grid, heuristic=h)
            if found.status != 'ok':
                raise ValueError(f'Q4 distance failed: {labels[i]}->{labels[j]}: {found.status}')
            row.append(found.cost)
            result.paths[f'{labels[i]}->{labels[j]}'] = [list(p) for p in found.path]
            result.searches += 1
            result.expanded += found.expanded
            result.generated += found.generated
            result.pair_stats.append(dict(source=labels[i], target=labels[j], cost=found.cost,
                                          expanded=found.expanded, elapsed_ns=found.elapsed_ns))
        result.matrix.append(row)
    result.elapsed_ns = perf_counter_ns()-begin
    return result


@dataclass
class GameResult:
    best_target: str | None = None
    best_value: int | None = None
    calls: int = 0
    expanded: int = 0
    leaf_evals: int = 0
    ab_cutoffs: int = 0
    tt_cutoffs: int = 0
    tt_probes: int = 0
    tt_hits: int = 0
    tt_entries: int = 0
    tt_capacity_skips: int = 0
    tie_researches: int = 0
    elapsed_ns: int = 0
    status: str = 'ok'

    def to_dict(self):
        return asdict(self)


def search_game(problem, depth, *, state=None, pruning=True, cache=False,
                ordering='fixed', eta=0.0, tt_capacity=200_000,
                seconds=60, max_calls=10_000_000):
    # 返回规范最优首步；并列按数字编号，不受访问顺序影响。
    # 访问数包含根和叶，缓存直接返回也计一次调用，但不计展开。
    # 窗口截断返回的是界；根遇到较小编号的疑似并列行动时完整窗口重搜。
    if type(depth) is not int or depth < 0 or depth > problem.count:
        raise ValueError('depth must lie between zero and the treasure count')
    if ordering not in ('fixed', 'immediate', 'competition') or not math.isfinite(eta) or eta < 0:
        raise ValueError('invalid action ordering')
    if tt_capacity < 0 or max_calls < 0 or seconds < 0:
        raise ValueError('invalid resource limit')
    state = state or problem.initial_state()
    if (state.current_player not in (0, 1) or not 0 <= state.remaining_mask < (1 << problem.count)
            or not 0 <= state.a_pos < problem.count+2 or not 0 <= state.b_pos < problem.count+2):
        raise ValueError('invalid game state')
    result, table = GameResult(), {}
    matrix, values, count = problem.distances, problem.values, problem.count
    begin = perf_counter_ns()

    def visit(a, b, mask, player, remaining, alpha, beta, root=False):
        if result.calls >= max_calls:
            raise OverflowError('node_limit')
        result.calls += 1
        if result.calls % 512 == 1 and perf_counter_ns()-begin > seconds*1e9:
            raise TimeoutError('timeout')
        if remaining == 0 or mask == 0:
            result.leaf_evals += 1
            return 0, None
        key = a, b, mask, player, remaining
        alpha0, beta0 = alpha, beta
        if cache:
            result.tt_probes += 1
            entry = table.get(key)
            if entry is not None:
                result.tt_hits += 1
                value, flag, action = entry
                if flag == 'EXACT':
                    return value, action
                if flag == 'LOWER':
                    alpha = max(alpha, value)
                else:
                    beta = min(beta, value)
                if alpha >= beta:
                    result.tt_cutoffs += 1
                    return value, action
        result.expanded += 1
        actions = [i for i in range(count) if mask & (1 << i)]
        own, other = (a, b) if player == 0 else (b, a)
        if ordering != 'fixed':
            def rank(i):
                own_gain = values[i]-matrix[own][i+2]
                competitive = eta*max(values[i]-matrix[other][i+2], 0) if ordering == 'competition' else 0
                return -(own_gain+competitive), i
            actions.sort(key=rank)
        best, best_action = (-math.inf if player == 0 else math.inf), None
        for index, target in enumerate(actions):
            gain = values[target]-matrix[own][target+2]
            delta = gain if player == 0 else -gain
            na, nb = (target+2, b) if player == 0 else (a, target+2)
            nm = mask ^ (1 << target)
            child, _ = visit(na, nb, nm, 1-player, remaining-1,
                             alpha-delta if pruning else -math.inf,
                             beta-delta if pruning else math.inf)
            candidate = delta+child
            # fail-low/high 结果只能说明界，不能直接宣称更小编号与当前最优并列。
            if root and best_action is not None and candidate == best and target < best_action and pruning:
                result.tie_researches += 1
                child, _ = visit(na, nb, nm, 1-player, remaining-1, -math.inf, math.inf)
                candidate = delta+child
            better = candidate > best if player == 0 else candidate < best
            if better or candidate == best and (best_action is None or target < best_action):
                best, best_action = candidate, target
            if pruning:
                if player == 0:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                if alpha >= beta:
                    if index+1 < len(actions):
                        result.ab_cutoffs += 1
                    break
        if cache:
            flag = 'UPPER' if best <= alpha0 else 'LOWER' if best >= beta0 else 'EXACT'
            if key in table or len(table) < tt_capacity:
                table[key] = best, flag, best_action
            else:
                result.tt_capacity_skips += 1
        return best, best_action

    try:
        value, action = visit(state.a_pos, state.b_pos, state.remaining_mask, state.current_player,
                              min(depth, state.remaining_mask.bit_count()), -math.inf, math.inf, True)
        result.best_value = state.utility+value
        result.best_target = problem.targets[action] if action is not None else None
    except (OverflowError, TimeoutError) as error:
        result.status = str(error)
        # 不把未完成搜索的当前值伪装成精确答案。
    result.tt_entries = len(table)
    result.elapsed_ns = perf_counter_ns()-begin
    return result


def diagnose_game(problem, depth):
    # 单独运行完整窗口诊断：所有首步的精确值与规范最优延续。
    # 本函数的重复求解成本不混入求一次最佳行动的主性能计时。
    initial = problem.initial_state()
    choices = {}
    for target in problem.actions(initial) if depth else []:
        state = problem.apply(initial, target)
        result = search_game(problem, depth-1, state=state, cache=True)
        if result.status != 'ok':
            raise RuntimeError(result.status)
        choices[problem.targets[target]] = result.best_value
    line, state = [], initial
    for remaining in range(depth, 0, -1):
        result = search_game(problem, remaining, state=state, cache=True)
        target = problem.targets.index(result.best_target)
        origin = state.a_pos if state.current_player == 0 else state.b_pos
        gain = problem.gain(origin, target)
        next_state = problem.apply(state, target)
        line.append(dict(turn=len(line)+1, player='A' if state.current_player == 0 else 'B',
                         origin=problem.labels[origin], target=result.best_target,
                         distance=problem.distances[origin][target+2], value=problem.values[target], gain=gain,
                         score_a=next_state.score_a, score_b=next_state.score_b, utility=next_state.utility))
        state = next_state
    best = max(choices.values()) if choices else initial.utility
    return dict(depth=depth, action_values=choices,
                all_optimal_actions=[t for t, v in choices.items() if v == best],
                best_value=best, line=line)
