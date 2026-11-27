"""Local smooth-response risks under supplied global derivative bounds.

The finite-radius test requires genuine derivative bounds on the complete
parameter interval. Bounds inferred only from samples do not certify it.
The parabolic formula is restricted to curvature * endpoint <= 2.
"""
import numpy as np
from scipy.optimize import brentq


def curvature_coefficient(velocity, acceleration):
    velocity = np.asarray(velocity, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    if velocity.ndim != 1 or acceleration.shape != velocity.shape:
        raise ValueError('Velocity and acceleration must be matching vectors.')
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(acceleration)):
        raise ValueError('Derivatives must be finite.')
    speed_squared = float(velocity @ velocity)
    if speed_squared <= 0:
        raise ValueError('The local law requires nonzero velocity.')
    normal = acceleration - velocity * float(velocity @ acceleration) / speed_squared
    return float(normal @ normal / 4)


def two_endpoint_recovery(curve, radius, speed_lower, speed_upper, acceleration_upper):
    values = np.array([radius, speed_lower, speed_upper, acceleration_upper], dtype=float)
    if not np.all(np.isfinite(values)) or radius <= 0 or speed_lower <= 0:
        raise ValueError('Use a positive radius, positive lower speed and finite bounds.')
    if speed_upper < speed_lower or acceleration_upper < 0:
        raise ValueError('Invalid derivative bounds.')
    margin = speed_lower**2 - 2 * radius * speed_upper * acceleration_upper
    if margin <= 0:
        raise ValueError('The supplied bounds do not certify the local regime.')
    left, right = np.asarray(curve(-radius), dtype=float), np.asarray(curve(radius), dtype=float)
    if left.ndim != 1 or left.shape != right.shape or not np.all(np.isfinite([left, right])):
        raise ValueError('Curve responses must be finite matching vectors.')
    midpoint, half_chord = (left + right) / 2, (right - left) / 2

    def balance(parameter):
        return float((np.asarray(curve(parameter)) - midpoint) @ half_chord)

    parameter = brentq(balance, -radius, radius, xtol=max(radius * 1e-14, 5e-324),
                        rtol=4 * np.finfo(float).eps)
    center = np.asarray(curve(parameter), dtype=float)
    mixed = float(half_chord @ half_chord)
    penalty = float((center - midpoint) @ (center - midpoint))
    return {'mixed_risk': mixed, 'single_risk': mixed + penalty, 'penalty': penalty,
            'mixed_center': midpoint.tolist(), 'single_center': center.tolist(),
            'parameter': float(parameter), 'regime_margin_lower': float(margin),
            'balance_residual': abs(balance(parameter))}


def hj_curvature_coefficient(hessian, third_directional, direction, horizon=1.0):
    direction = np.asarray(direction, dtype=float)
    hessian = np.asarray(hessian, dtype=float)
    third_directional = np.asarray(third_directional, dtype=float)
    if horizon <= 0 or hessian.shape != (len(direction), len(direction)):
        raise ValueError('Use a positive horizon and matching Hessian.')
    velocity = direction - horizon * hessian @ direction
    return curvature_coefficient(velocity, -horizon * third_directional)


def parabolic_recovery(curvature, radius):
    if not np.isfinite(curvature) or not np.isfinite(radius) or curvature < 0 or radius <= 0:
        raise ValueError('Use nonnegative curvature and a positive finite radius.')
    endpoint = brentq(lambda b: b + curvature**2 * b**3 / 2 - radius, 0, radius,
                      xtol=max(radius * 1e-14, 5e-324))
    if curvature * endpoint > 2 + 1e-12:
        raise ValueError('The parabolic risk formula requires curvature * endpoint <= 2.')
    mixed = endpoint**2
    penalty = curvature**2 * endpoint**4 / 4
    energy = ((radius - endpoint)**2 + penalty) / 2
    return {'endpoint': float(endpoint), 'mixed_risk': float(mixed),
            'single_risk': float(mixed + penalty), 'penalty': float(penalty),
            'hj_energy': float(energy)}


def disk_recovery(height, radius):
    if not np.isfinite(height) or not np.isfinite(radius) or height <= 1 or radius <= 0:
        raise ValueError('The smooth HJ example uses exterior height > 1 and radius > 0.')
    length = np.hypot(height, radius)
    sagitta = radius**2 / (length * (length + height))
    return {'mixed_risk': float(radius**2 / length**2), 'single_risk': float(2 * sagitta),
            'penalty': float(sagitta**2), 'predicted_coefficient': float(1 / (4 * height**4))}
