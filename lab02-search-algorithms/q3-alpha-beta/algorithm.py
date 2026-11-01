# 题目三接口：Minimax、Alpha-Beta和重复状态缓存。
import json
from pathlib import Path
from searchlab.fixtures import teacher_tree
from searchlab.tree import TreeNode, minimax, alpha_beta, search_tree, tree_from_dict

def solve(data, *, algorithm='AB', ordering='fixed', cache=False, **options):
    root = tree_from_dict(data)
    return search_tree(root, pruning=algorithm == 'AB', ordering=ordering, cache=cache, record=True).to_dict()

def sample():
    # 使用给定建树函数保留共享max2对象；JSON树按边展开的回传值一致。
    return {name: search_tree(teacher_tree(), pruning=name == 'AB', record=True).to_dict() for name in ['MM', 'AB']}
