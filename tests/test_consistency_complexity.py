"""Independent analytic controls for the computational separation construction."""
from fractions import Fraction
from itertools import product
import unittest

import numpy as np

from src.consistency_complexity import (boolean_solutions, central_risk,
    clause_matrix, response, solve_consistent_center, witnesses)


class ConsistencyComplexityTests(unittest.TestCase):
    def test_clause_reduction_including_repeated_variables(self):
        for count, clauses in [(3, [(0, 1, 2)]), (1, [(0, 0, 0)]),
                               (2, [(0, 0, 1), (0, 1, 1)])]:
            matrix = clause_matrix(count, clauses)
            for bits in product((0, 1), repeat=count):
                sigma = np.r_[1, 2*np.asarray(bits)-1]
                self.assertEqual(bool(np.all(matrix @ sigma == 0)),
                                 all(sum(bits[i] for i in row) == 1 for row in clauses))

    def test_known_satisfiable_instance_attains_exact_baseline(self):
        matrix = clause_matrix(3, [(0, 1, 2)])
        result = solve_consistent_center(matrix)
        self.assertAlmostEqual(result['proper_upper'], 6.5, places=7)
        self.assertAlmostEqual(result['proper_lower'], 6.5, places=7)
        self.assertAlmostEqual(result['mixed_risk'], 4.)

    def test_unsatisfiable_repeated_clause_has_exact_one_dimensional_value(self):
        # u_ref=-3*u_1, |u_ref|<=1/4; the risk decreases up to the endpoint.
        result = solve_consistent_center(clause_matrix(1, [(0, 0, 0)]))
        self.assertAlmostEqual(result['proper_upper'], float(Fraction(125, 36)), places=7)
        self.assertGreater(result['proper_lower']-result['baseline'], 1/6)

    def test_distinct_clause_unsatisfiable_control(self):
        clauses = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
        self.assertEqual(boolean_solutions(4, clauses), [])
        result = solve_consistent_center(clause_matrix(4, clauses))
        self.assertAlmostEqual(result['proper_upper'], float(Fraction(649, 72)), places=7)

    def test_full_image_ball_and_two_witness_dominance(self):
        rng = np.random.default_rng(20261115)
        for k in [2, 5, 9]:
            n = 2*k
            plus, minus = witnesses(k)
            center = np.full(2*n, .5)
            for _ in range(120):
                u = rng.uniform(-.25, .25, k)
                candidate = response(0, u)
                exact = max(np.sum((candidate-plus)**2), np.sum((candidate-minus)**2))
                self.assertAlmostEqual(exact, central_risk(u), places=10)
                t = rng.uniform(-1, 1)
                other = response(t, (1-abs(t))*rng.uniform(-.25, .25, k))
                self.assertLessEqual(np.sum((other-center)**2), n/2+1e-10)
                self.assertLessEqual(np.sum((other-candidate)**2), exact+1e-10)
                biased = response(t, (1-abs(t))*u)
                witness_risk = max(np.sum((biased-plus)**2), np.sum((biased-minus)**2))
                self.assertGreaterEqual(witness_risk, central_risk(u)+n*t*t-1e-10)

    def test_no_source_restriction_has_closed_form(self):
        result = solve_consistent_center(np.empty((0, 3)))
        self.assertAlmostEqual(result['proper_upper'], 39/8, places=7)
        self.assertAlmostEqual(result['proper_lower'], 39/8, places=7)


if __name__ == '__main__':
    unittest.main()
