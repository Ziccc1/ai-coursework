import math
import random
import unittest
from searchlab.fixtures import MAZE, WEIGHTED, teacher_tree
from searchlab.grid import Grid, astar, astar_maze, astar_weighted, build_heuristic
from searchlab.reference import check_route, reference_tree, shortest_distances
from searchlab.tree import TreeNode, alpha_beta, exact_values, search_tree, tree_from_dict

from algorithm import solve, sample

class QuestionTests(unittest.TestCase):
    def test_teacher(self):
        results = sample()
        self.assertTrue(all(r['status'] == 'ok' and r['cost'] == 7 for r in results.values()))

    def test_random_against_reference(self):
        rng = random.Random(41020)
        for _ in range(80):
            data = [[None if rng.random() < .25 else rng.choice([0,1,2,8]) for _ in range(7)] for _ in range(6)]
            data[0][0] = data[-1][-1] = 1
            grid = Grid(data, (0,0), (5,6), 'weighted')
            expected = shortest_distances(grid.costs, grid.start).get(grid.goal, -1)
            truth = shortest_distances(grid.costs, grid.goal, reverse=True)
            for kind in ['zero','corrected','relaxed']:
                h = build_heuristic(grid, kind)[0]
                self.assertEqual(h(grid.goal), 0)
                for node, remaining in truth.items():
                    self.assertLessEqual(h(node), remaining)
                    for nxt, weight in grid.neighbors(node):
                        self.assertLessEqual(h(node), weight+h(nxt))
                for tie in ['larger_g', 'smaller_g', 'fifo']:
                    check_route(grid, astar(grid, heuristic=h, tie_break=tie), expected)

    def test_boundaries(self):
        self.assertEqual(solve(dict(grid=[[1,None,1]], start=[0,0], goal=[0,2]))['status'], 'unreachable')
        self.assertEqual(solve(dict(grid=[[1]], start=[0,0], goal=[0,0]))['cost'], 0)
        self.assertEqual(solve(dict(grid=[['S','E']], start=[0,0], goal=[0,1]))['cost'], 0)
