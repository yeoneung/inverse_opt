"""Independent analytic and exhaustive checks of the new computational study."""
import unittest
import cvxpy as cp
import numpy as np

from src.multidimensional_certification import (
    Projector, SETTINGS, all_regions_reference, query_certificate,
    sampled_ball, structured_certificate)


class MultidimensionalCertificationTests(unittest.TestCase):
    def test_balanced_dimension_sweep_keeps_target_and_existing_columns_fixed(self):
        from experiments.balanced_capacity import instance
        baseline = instance(32, 8, 846100, .7, .75)
        for p in [2,4,8]:
            A,b,c,G,_ = instance(32,p,846100,.7,.75)
            for actual,expected in zip([A,b,c],baseline[:3]):
                np.testing.assert_array_equal(actual,expected)
            np.testing.assert_allclose(G,baseline[3][:,:p],atol=1e-14)
            np.testing.assert_allclose(np.linalg.svd(G,compute_uv=False),.75,atol=1e-12)

    def test_reusable_projection_matches_independent_cvxpy(self):
        A = np.array([[-1., 0.], [0., -1.], [1., 1.]])
        b = np.array([0., 0., 1.])
        projector = Projector(A, b, 2)
        for z in [np.array([2., .7]), np.array([-.5, .2]), np.array([.1, .3])]:
            actual, _ = projector(z)
            x = cp.Variable(2)
            problem = cp.Problem(cp.Minimize(cp.sum_squares(x-z)), [A @ x <= b])
            problem.solve(**SETTINGS)
            np.testing.assert_allclose(actual, x.value, atol=2e-6)

    def test_contact_mean_is_feasible_and_variance_is_a_lower_value(self):
        points = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -.5]])
        ball = sampled_ball(points)
        np.testing.assert_allclose(ball['anchor'], np.zeros(2), atol=2e-5)
        self.assertLessEqual(ball['variance'], 1.)
        self.assertAlmostEqual(ball['sample_upper'], 1., places=5)
        self.assertTrue((ball['weights'] >= 0).all())
        self.assertAlmostEqual(float(ball['weights'].sum()), 1.)

    def test_global_reference_for_two_independent_folds(self):
        G = np.vstack((np.eye(2), -np.eye(2)))
        reference = all_regions_reference(-np.eye(4), np.zeros(4), np.zeros(4), G)
        self.assertEqual(reference['regions'], 4)
        self.assertAlmostEqual(reference['R_lower'], 1., places=6)
        self.assertAlmostEqual(reference['R_upper'], 1., places=6)
        self.assertAlmostEqual(reference['P_lower'], 2., places=5)
        self.assertAlmostEqual(reference['P_upper'], 2., places=5)

    def test_no_reference_information_is_needed_for_a_positive_gap(self):
        G = np.array([[1.], [-1.]])
        result = query_certificate(-np.eye(2), np.zeros(2), np.zeros(2), G, 17)
        self.assertGreater(result['adaptive_gap'], .24)
        self.assertLessEqual(result['adaptive_gap'], .5+1e-7)
        self.assertGreaterEqual(result['upper'], .5-1e-7)
        self.assertGreaterEqual(result['plain_upper'], result['upper'])
        self.assertGreater(result['affine_leaves'], 0)
        self.assertGreaterEqual(result['adaptive_total_seconds'],
            result['forward_seconds']+result['anchor_seconds']+result['upper_seconds']+result['adaptive_seconds'])

    def test_multidimensional_query_upper_covers_the_entire_reference_image(self):
        G = np.vstack((np.eye(2), -np.eye(2)))
        reference = all_regions_reference(-np.eye(4), np.zeros(4), np.zeros(4), G)
        result = query_certificate(-np.eye(4), np.zeros(4), np.zeros(4), G, 65)
        exact_upper = np.max(np.sum((reference['image_vertices']-result['anchor'])**2, axis=1))
        self.assertGreaterEqual(result['upper']+1e-7, exact_upper)
        self.assertLessEqual(result['adaptive_gap'], 1.+1e-7)

    def test_dense_rotations_preserve_known_ball_risks(self):
        for p in [2, 4]:
            result = structured_certificate(p, 817340+p)
            self.assertAlmostEqual(result['R_reference'], 1-1/(2*p))
            self.assertAlmostEqual(result['P_reference'], 1.)
            self.assertGreater(result['adaptive_gap'], 0.)
            self.assertLessEqual(result['adaptive_gap'], result['penalty_reference']+1e-7)
            self.assertGreaterEqual(result['upper']+1e-7, result['R_reference'])

    def test_affine_ball_control_has_no_false_positive(self):
        result = structured_certificate(3, 9417, affine_control=True)
        self.assertLessEqual(result['original_gap'], 1e-8)
        self.assertLessEqual(result['adaptive_gap'], 1e-8)


if __name__ == '__main__':
    unittest.main()
