# 固定种子实例生成与可重建序列化。
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path
import random

from .reference import bfs_route
from .tree import TreeNode


def local_random(family, seed, **params):
    raw = json.dumps(['v1', family, seed, params], sort_keys=True).encode()
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big'))


def random_grid(size, density, seed, family='grid', connected=True):
    rng = local_random(family, seed, size=size, density=density)
    for attempt in range(1, 1001):
        costs = [[None if rng.random() < density else 1 for _ in range(size)] for _ in range(size)]
        costs[0][0] = costs[-1][-1] = 1
        if not connected or bfs_route(costs, (0, 0), (size-1, size-1)):
            return costs, attempt
    raise RuntimeError('generation_failed: no connected grid in 1000 attempts')


def structure_grid(kind, size=30):
    grid = [[None]*size for _ in range(size)]
    start, goal = (0, 0), (size-1, size-1)
    if kind == 'open':
        grid = [[1]*size for _ in range(size)]
    elif kind == 'serpentine':
        for r in range(size-1):
            if r % 2 == 0:
                grid[r] = [1]*size
            else:
                grid[r][size-1 if (r//2) % 2 == 0 else 0] = 1
        grid[-1][-1] = 1
    elif kind == 'comb':
        for r in range(size):
            grid[r][1] = 1
            if r % 2 == 0:
                for c in range(1, size-1):
                    grid[r][c] = 1
        for c in range(1, size):
            grid[-1][c] = 1
        start = 0, 1
    elif kind == 'detour':
        grid = [[1]*size for _ in range(size)]
        for r in range(size-1):
            grid[r][size//2] = None
        start, goal = (0, size//2-1), (0, size//2+1)
    else:
        raise ValueError(kind)
    return grid, start, goal


def weighted_grid(size, density, seed, pattern='mild', family='weighted'):
    data, attempts = random_grid(size, density, seed, family, connected=True)
    rng = local_random(family+'weights', seed, size=size, density=density, pattern=pattern)
    for r in range(size):
        for c in range(size):
            if data[r][c] is None:
                continue
            if pattern == 'uniform':
                value = 1
            elif pattern == 'mild':
                value = rng.choices([1, 2, 3], [.6, .3, .1])[0]
            elif pattern == 'sparse':
                value = rng.choices([1, 9], [.9, .1])[0]
            elif pattern == 'barrier':
                value = 9 if c == size//2 and r != size-1 else 1
            elif pattern == 'corridor':
                value = 1 if r == 0 or c == size-1 else 5
            else:
                raise ValueError(pattern)
            data[r][c] = value
    data[0][0] = 1
    return data, attempts


def random_tree(branching, depth, seed, early_stop=0.0):
    rng = local_random('tree', seed, branching=branching, depth=depth, early_stop=early_stop)

    def build(level):
        if level == depth or level > 0 and rng.random() < early_stop:
            return TreeNode(value=rng.randint(-20, 20))
        return TreeNode(children=[build(level+1) for _ in range(branching)])

    return build(0)


def random_dag(branching, depth, seed, reuse):
    rng = local_random('dag', seed, branching=branching, depth=depth, reuse=reuse)
    layers = [[] for _ in range(depth+1)]

    def build(level):
        if layers[level] and rng.random() < reuse:
            return rng.choice(layers[level])
        node = TreeNode()
        layers[level].append(node)
        if level == depth:
            node.value = rng.randint(-20, 20)
        else:
            node.children = [build(level+1) for _ in range(branching)]
        return node

    return build(0)


def tree_payload(root):
    queue, index, nodes = [root], {id(root): 0}, []
    for node in queue:
        children = []
        for child in node.children:
            if id(child) not in index:
                index[id(child)] = len(queue)
                queue.append(child)
            children.append(index[id(child)])
        nodes.append({'value': node.value, 'children': children})
    return {'kind': 'tree', 'root': 0, 'nodes': nodes}


def load_tree(payload):
    nodes = [TreeNode(value=n['value']) for n in payload['nodes']]
    for node, data in zip(nodes, payload['nodes']):
        node.children = [nodes[i] for i in data['children']]
    return nodes[payload['root']]


def save_instance(directory, payload):
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    digest = hashlib.sha256(raw).hexdigest()
    path = Path(directory) / (digest + '.json.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(gzip.compress(raw, mtime=0))
    return digest, len(raw)
