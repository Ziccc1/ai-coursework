import json
from pathlib import Path
import random
import unittest

from searchlab.treasure import (GameState, TreasureProblem, build_distances,
                               diagnose_game, search_game)
from searchlab.treasure_reference import enumerate_game, reference_distances, reference_game
from searchlab.treasure_generators import random_case, scaled_cases, subset_case

ROOT = Path(__file__).resolve().parents[1]


def matrix_problem(n, seed):
    rng = random.Random(seed)
    case = dict(grid=[[1]*(n+2)], points={'A': [0, 0], 'B': [0, 1]},
                values={f'T{i+1}': rng.randint(-3, 25) for i in range(n)})
    case['points'].update({f'T{i+1}': [0, i+2] for i in range(n)})
    # 不预设三角关系，强化对搜索递推和边界的检验；寻路正确性另测。
    matrix = [[0 if i == j else rng.randint(1, 18) for j in range(n+2)] for i in range(n+2)]
    return TreasureProblem(case, matrix)


class TreasureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((ROOT/'设计材料/第四题案例定义.json').read_text(encoding='utf-8'))['cases']
        cls.expected = {r['id']: r for r in json.loads((ROOT/'设计材料/第四题案例核验结果.json').read_text(encoding='utf-8'))['cases']}

    def test_astar_all_curated_pairs_and_trace(self):
        for case in self.cases:
            result = build_distances(case)
            self.assertEqual(result.matrix, reference_distances(case))
            self.assertEqual(result.matrix, self.expected[case['id']]['distance_matrix'])
            self.assertEqual(result.searches, 30)
            for key, path in result.paths.items():
                a, b = key.split('->')
                self.assertEqual(path[0], case['points'][a])
                self.assertEqual(path[-1], case['points'][b])
                self.assertTrue(all(abs(u[0]-v[0])+abs(u[1]-v[1]) == 1 for u, v in zip(path, path[1:])))
                self.assertEqual(sum(case['grid'][r][c] for r, c in path[1:]), result.matrix[result.labels.index(a)][result.labels.index(b)])

    def test_curated_depths_and_every_algorithm(self):
        for case in self.cases:
            problem = TreasureProblem(case, reference_distances(case))
            for depth in range(1, 5):
                ref = self.expected[case['id']]['by_depth'][str(depth)]
                exhaustive, actions = enumerate_game(problem.distances, problem.values, depth)
                self.assertEqual(exhaustive, ref['best_value'])
                for prune, cache, order in [(False, False, 'fixed'), (True, False, 'fixed'),
                                             (True, False, 'immediate'), (True, True, 'fixed'),
                                             (True, True, 'competition')]:
                    got = search_game(problem, depth, pruning=prune, cache=cache, ordering=order, eta=1)
                    self.assertEqual((got.best_target, got.best_value), (ref['best_target'], ref['best_value']))
                    self.assertEqual(got.status, 'ok')
                diagnostic = diagnose_game(problem, depth)
                self.assertEqual(diagnostic['action_values'], ref['action_values'])
                self.assertEqual(diagnostic['line'][-1]['utility'], ref['best_value'])

    def test_random_search_bounds_and_canonical_ties(self):
        for seed in range(24):
            problem = matrix_problem(6, seed)
            for depth in [2, 4, 6]:
                ref = reference_game(problem.distances, problem.values, depth)
                for prune, cache, order in [(False, False, 'fixed'), (True, False, 'immediate'),
                                             (True, True, 'fixed'), (True, True, 'competition')]:
                    got = search_game(problem, depth, pruning=prune, cache=cache, ordering=order, eta=.5)
                    self.assertEqual(got.best_value, ref['value'], (seed, depth, cache, order))
                    self.assertEqual(got.best_target, problem.targets[min(ref['optimal_actions'])], (seed, depth, cache, order))

    def test_history_offset_min_player_and_small_cache(self):
        problem = matrix_problem(7, 92)
        for who in [0, 1]:
            state = GameState(2, 3, 0b1111100, 37, -12, who)
            ref = reference_game(problem.distances, problem.values, 5, a=2, b=3, mask=state.remaining_mask, player=who)
            for capacity in [0, 3, 200000]:
                got = search_game(problem, 5, state=state, cache=True, ordering='competition', eta=2, tt_capacity=capacity)
                self.assertEqual(got.best_value, 49+ref['value'])
                self.assertEqual(got.best_target, problem.targets[min(ref['optimal_actions'])])

    def test_negative_gains_mandatory_moves_and_limits(self):
        case = self.cases[0]
        case = {**case, 'values': {t: 0 for t in case['values']}}
        problem = TreasureProblem(case, reference_distances(case))
        got = search_game(problem, 4, cache=True, ordering='immediate')
        ref = reference_game(problem.distances, problem.values, 4)
        self.assertIsNotNone(got.best_target)
        self.assertEqual(got.best_value, ref['value'])
        for options, status in [({'max_calls': 1}, 'node_limit'), ({'seconds': 0}, 'timeout')]:
            result = search_game(problem, 4, **options)
            self.assertEqual(result.status, status)
            self.assertIsNone(result.best_value)
            self.assertIsNone(result.best_target)
        self.assertEqual(search_game(problem, 0).best_value, 0)
        self.assertEqual(search_game(problem, 0).calls, 1)

    def test_unit_entering_cost_and_input_validation(self):
        problem = matrix_problem(4, 1)
        result = build_distances(problem.case)
        self.assertEqual(result.matrix[0][2], 2)  # 若误用题目二的免费终点规则，此处会错误地得到1。
        state = problem.initial_state()
        next_state = problem.apply(state, 0)
        self.assertEqual(state.remaining_mask, 15)
        self.assertEqual(next_state.remaining_mask, 14)
        with self.assertRaises(ValueError):
            problem.apply(next_state, 0)
        bad = {**problem.case, 'grid': [[1, 1, 2, 1, 1, 1]]}
        with self.assertRaises(ValueError):
            build_distances(bad)
        bad = {**problem.case, 'grid': [[1, 1, None, 1, 1, 1]]}
        with self.assertRaises(ValueError):
            build_distances(bad)
        for depth in [-1, 5, True]:
            with self.assertRaises(ValueError):
                search_game(problem, depth)

    def test_controlled_generators_and_random_real_maps(self):
        for seed in [10, 13, 29]:
            case = random_case(15, 8, seed, 'unit_test')
            self.assertEqual(case, random_case(15, 8, seed, 'unit_test'))
            distances = build_distances(case)
            self.assertEqual(distances.matrix, reference_distances(case))
            problem = TreasureProblem(case, distances.matrix)
            ref = reference_game(problem.distances, problem.values, 6)
            result = search_game(problem, 6, cache=True, ordering='competition', eta=2)
            self.assertEqual(result.best_value, ref['value'])
            prefix = subset_case(case, 4)
            self.assertEqual(prefix['grid'], case['grid'])
            self.assertEqual(prefix['points']['A'], case['points']['A'])
            generated = scaled_cases(seed)
            self.assertEqual(generated, scaled_cases(seed))
            for scaled in generated:
                self.assertTrue(all(x is None or x == 1 for row in scaled['grid'] for x in row))
                self.assertTrue(all(v >= 0 for row in reference_distances(scaled) for v in row))


if __name__ == '__main__':
    unittest.main()
