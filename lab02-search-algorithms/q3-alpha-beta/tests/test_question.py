import math
import random
import unittest
from searchlab.fixtures import MAZE, WEIGHTED, teacher_tree
from searchlab.grid import Grid, astar, astar_maze, astar_weighted, build_heuristic
from searchlab.reference import check_route, reference_tree, shortest_distances
from searchlab.tree import TreeNode, alpha_beta, exact_values, search_tree, tree_from_dict

class TreeTests(unittest.TestCase):
    def test_teacher_and_shared_nodes(self):
        root = teacher_tree()
        self.assertIs(root.children[0].children[1], root.children[1].children[0])
        mm = search_tree(root, pruning=False, record=True)
        ab = search_tree(root, pruning=True, record=True)
        self.assertEqual((mm.value, ab.value), (5, 5))
        self.assertEqual((mm.calls, mm.leaf_evals), (15, 8))
        self.assertEqual((ab.calls, ab.leaf_evals, ab.ab_cutoffs), (14, 7, 1))

    def test_random_graphs_all_modes(self):
        rng = random.Random(19)
        for _ in range(60):
            layer = [TreeNode(value=rng.randint(-20, 20)) for _ in range(8)]
            for depth in range(4):
                layer = [TreeNode(children=rng.sample(layer, min(len(layer), rng.randint(1, 4)))) for _ in range(max(1, 7-depth))]
            root = TreeNode(children=layer)
            expected = reference_tree(root)
            oracle = exact_values(root)
            for pruning in [False, True]:
                for cache in [False, True]:
                    for order in ['fixed', 'reverse', 'random', 'oracle_good', 'oracle_bad']:
                        result = search_tree(root, pruning=pruning, cache=cache, ordering=order, seed=18, oracle=oracle)
                        self.assertEqual(result.value, expected)

    def test_zero_negative_single_and_invalid(self):
        self.assertEqual(alpha_beta(TreeNode(value=0)), 0)
        self.assertEqual(alpha_beta(TreeNode(children=[TreeNode(-9), TreeNode(-2)])), -2)
        for root in [TreeNode(), TreeNode(value=math.nan), TreeNode(1, [TreeNode(2)])]:
            with self.assertRaises(ValueError):
                alpha_beta(root)
        cycle = TreeNode()
        cycle.children = [cycle]
        with self.assertRaises(ValueError):
            alpha_beta(cycle)
        with self.assertRaises(ValueError):
            tree_from_dict({'is_maximizing_player': False, 'value': 1})

    def test_node_limit(self):
        self.assertEqual(search_tree(teacher_tree(), max_calls=1).status, 'node_limit')
