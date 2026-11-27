"""Geometric and HJ checks of the added local curvature results."""
from fractions import Fraction
import unittest

import numpy as np
from scipy.optimize import minimize_scalar

from src.smooth_curvature import (curvature_coefficient, disk_recovery,
                                 hj_curvature_coefficient, parabolic_recovery,
                                 two_endpoint_recovery)


class SmoothCurvatureTests(unittest.TestCase):
    def test_nonlinear_speed_on_a_line_has_no_penalty(self):
        direction = np.array([1.0, 2.0])
        curve = lambda t: (t + 0.3 * t**2) * direction
        result = two_endpoint_recovery(curve, 0.1, 2.0, 2.5, 1.4)
        self.assertLess(result['penalty'], 1e-26)
        self.assertGreater(abs(result['parameter']), 1e-4)

    def test_asymmetric_curve_matches_full_sample_minimax(self):
        curve = lambda t: np.array([t + 0.4 * t**2, 0.7 * t**2 + 0.3 * t**3])
        radius = 0.05
        second_bound = np.linalg.norm([0.8, 1.4]) + 1.8 * radius
        result = two_endpoint_recovery(curve, radius, 1 - radius * second_bound,
                                       1 + radius * second_bound, second_bound)
        points = np.array([curve(t) for t in np.linspace(-radius, radius, 2001)])
        risk = lambda t: np.max(np.sum((points - curve(t))**2, axis=1))
        direct = minimize_scalar(risk, bounds=(-radius, radius), method='bounded',
                                 options={'xatol': 1e-15})
        self.assertLess(abs(direct.fun - result['single_risk']), 1e-10)
        self.assertLess(abs(result['penalty'] / radius**4 - 0.49), 5e-4)

    def test_parabolic_price_matches_rational_energy_identity(self):
        curvature, endpoint = Fraction(2), Fraction(1, 10)
        radius = endpoint + curvature**2 * endpoint**3 / 2
        expected = curvature**2 * endpoint**4 / 4
        result = parabolic_recovery(float(curvature), float(radius))
        self.assertAlmostEqual(result['penalty'], float(expected), places=14)
        self.assertAlmostEqual(result['penalty'], 2 * result['hj_energy'] /
                               (1 + float(curvature)**2 * result['endpoint']**2), places=14)
        points = np.array([[x, -float(curvature) * x**2 / 2]
                           for x in np.linspace(-float(endpoint), float(endpoint), 401)])
        grid_risks = np.max(np.sum((points[:, None, :] - points[None, :, :])**2, axis=2), axis=1)
        self.assertAlmostEqual(grid_risks.min(), result['single_risk'], places=13)

    def test_hj_disk_derivatives_give_the_curvature_coefficient(self):
        height, step = 2.0, 1e-3
        point, direction = np.array([0.0, height]), np.array([1.0, 0.0])
        for horizon in [0.5, 1.0, 2.0]:
            gradient = lambda z: (z - z / np.linalg.norm(z)) / horizon
            hessian = np.diag([1 - 1 / height, 1.0]) / horizon
            third = (gradient(point + step * direction) - 2 * gradient(point) +
                     gradient(point - step * direction)) / step**2
            coefficient = hj_curvature_coefficient(hessian, third, direction, horizon)
            self.assertLess(abs(coefficient - 1 / 64), 2e-8)
        observed = disk_recovery(height, 0.025)
        self.assertLess(abs(observed['penalty'] / 0.025**4 - 1 / 64), 5e-6)

    def test_rigid_motion_and_tangential_acceleration(self):
        velocity, acceleration = np.array([1.0, 0.0, 0.0]), np.array([0.8, 1.4, -0.5])
        rotation, _ = np.linalg.qr(np.random.default_rng(17).normal(size=(3, 3)))
        expected = curvature_coefficient(velocity, acceleration)
        self.assertAlmostEqual(curvature_coefficient(rotation @ velocity,
                                                    rotation @ acceleration), expected)
        self.assertAlmostEqual(curvature_coefficient(velocity, acceleration + 7 * velocity), expected)

    def test_inapplicable_regimes_are_rejected(self):
        with self.assertRaises(ValueError):
            curvature_coefficient([0.0, 0.0], [1.0, 0.0])
        with self.assertRaises(ValueError):
            two_endpoint_recovery(lambda t: np.array([t, t**2]), 1.0, 0.5, 3.0, 2.0)
        with self.assertRaises(ValueError):
            parabolic_recovery(1.0, 10.0)
        boundary = parabolic_recovery(1.0, 6.0)
        self.assertAlmostEqual(boundary['mixed_risk'], 4.0)
        self.assertAlmostEqual(boundary['single_risk'], 8.0)


if __name__ == '__main__':
    unittest.main()
