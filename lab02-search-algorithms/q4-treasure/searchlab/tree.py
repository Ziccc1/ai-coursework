# Minimax / Alpha-beta，支持共享子树与有界置换表。
from dataclasses import asdict, dataclass, field
import math
import random
from time import perf_counter_ns


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []


@dataclass
class TreeResult:
    value: int | float | None = None
    calls: int = 0
    expanded: int = 0
    leaf_evals: int = 0
    ab_cutoffs: int = 0
    tt_cutoffs: int = 0
    tt_probes: int = 0
    tt_hits: int = 0
    tt_entries: int = 0
    elapsed_ns: int = 0
    status: str = 'ok'
    visited_leaves: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def tree_from_dict(data, is_max=True):
    if 'tree_structure' in data:
        data = data['tree_structure']
    if 'is_maximizing_player' in data and data['is_maximizing_player'] != is_max:
        raise ValueError('player labels must alternate from MAX root')
    children = data.get('children', [])
    return TreeNode(data.get('value'), [tree_from_dict(c, not is_max) for c in children])


def validate_tree(root):
    active, done = set(), set()

    def check(node):
        key = id(node)
        if key in active:
            raise ValueError('cycles are not valid game trees/DAGs')
        if key in done:
            return
        active.add(key)
        if node.children:
            if node.value is not None:
                raise ValueError('internal nodes must not have a value')
            for child in node.children:
                check(child)
        elif isinstance(node.value, bool) or not isinstance(node.value, (int, float)) or not math.isfinite(node.value):
            raise ValueError('leaf value must be a finite number')
        active.remove(key)
        done.add(key)

    check(root)
    return len(done)


def exact_values(root):
    # 排序 oracle 的完整求值；与主搜索耗时分开披露。
    values = {}

    def visit(node, maximizing):
        key = id(node), maximizing
        if key not in values:
            values[key] = (max if maximizing else min)(visit(c, not maximizing) for c in node.children) if node.children else node.value
        return values[key]

    visit(root, True)
    return values


def search_tree(root, *, pruning=True, cache=False, ordering='fixed', seed=0,
                oracle=None, record=False, max_calls=10_000_000, seconds=60,
                tt_capacity=200_000, validate=True):
    if validate:
        validate_tree(root)
    if ordering not in ('fixed', 'reverse', 'random', 'oracle_good', 'oracle_bad'):
        raise ValueError('invalid tree ordering')
    if ordering.startswith('oracle') and oracle is None:
        raise ValueError('oracle ordering requires separately prepared exact values')
    rng, table = random.Random(seed), {}
    result = TreeResult()
    begin = perf_counter_ns()

    def visit(node, maximizing, alpha, beta):
        result.calls += 1
        if result.calls > max_calls:
            raise OverflowError('node_limit')
        if result.calls % 512 == 1 and perf_counter_ns() - begin > seconds * 1e9:
            raise TimeoutError('timeout')
        key = id(node), maximizing
        original_alpha, original_beta = alpha, beta
        if cache:
            result.tt_probes += 1
            entry = table.get(key)
            if entry is not None:
                result.tt_hits += 1
                value, flag = entry
                if flag == 'EXACT':
                    return value
                if flag == 'LOWER':
                    alpha = max(alpha, value)
                else:
                    beta = min(beta, value)
                if pruning and alpha >= beta:
                    result.tt_cutoffs += 1
                    return value
        if not node.children:
            result.leaf_evals += 1
            if record:
                result.visited_leaves.append(node.value)
            value = node.value
        else:
            result.expanded += 1
            children = list(node.children)
            if ordering == 'reverse':
                children.reverse()
            elif ordering == 'random':
                rng.shuffle(children)
            elif ordering.startswith('oracle'):
                reverse = maximizing if ordering == 'oracle_good' else not maximizing
                children.sort(key=lambda c: oracle[id(c), not maximizing], reverse=reverse)
            value = -math.inf if maximizing else math.inf
            for i, child in enumerate(children):
                child_value = visit(child, not maximizing, alpha, beta)
                if maximizing:
                    value, alpha = max(value, child_value), max(alpha, child_value)
                else:
                    value, beta = min(value, child_value), min(beta, child_value)
                if pruning and alpha >= beta:
                    if i + 1 < len(children):
                        result.ab_cutoffs += 1
                    break
        if cache and (key in table or len(table) < tt_capacity):
            flag = 'EXACT'
            if pruning:
                flag = 'UPPER' if value <= original_alpha else 'LOWER' if value >= original_beta else 'EXACT'
            table[key] = value, flag
        return value

    try:
        result.value = visit(root, True, -math.inf, math.inf)
    except (OverflowError, TimeoutError) as exc:
        result.status = str(exc)
    result.elapsed_ns = perf_counter_ns() - begin
    result.tt_entries = len(table)
    return result


def minimax(root):
    return search_tree(root, pruning=False).value


def alpha_beta(root):
    return search_tree(root, pruning=True).value
