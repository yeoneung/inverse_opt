"""Exact reduction and exhaustive convex solves for objective consistency.

This is a small-instance verifier, not a polynomial-time minimax algorithm.
The target map is coordinatewise projection onto one fixed orthant.
"""
from itertools import product
import time

import cvxpy as cp
import numpy as np


def clause_matrix(variable_count, clauses):
    """Column zero is the reference sign; clause variables use zero-based indices."""
    if variable_count < 1:
        raise ValueError('At least one Boolean variable is required.')
    result = np.zeros((len(clauses), variable_count + 1), dtype=int)
    for row, clause in enumerate(clauses):
        if len(clause) != 3 or any(i < 0 or i >= variable_count for i in clause):
            raise ValueError('Each clause must contain three valid variable indices.')
        result[row, 0] = 1
        for index in clause:
            result[row, index + 1] += 1
    return result


def boolean_solutions(variable_count, clauses):
    return [bits for bits in product((0, 1), repeat=variable_count)
            if all(sum(bits[i] for i in clause) == 1 for clause in clauses)]


def response(t, u):
    u = np.asarray(u, dtype=float)
    latent = t + np.r_[u, -u]
    return np.r_[np.maximum(latent, 0), np.maximum(-latent, 0)]


def witnesses(k):
    n = 2 * k
    return np.r_[np.ones(n), np.zeros(n)], np.r_[np.zeros(n), np.ones(n)]


def central_risk(u):
    u = np.asarray(u, dtype=float)
    return float(2 * len(u) + 2 * (u @ u) - 2 * np.linalg.norm(u, 1))


def solve_consistent_center(matrix, max_patterns=4096):
    """Enumerate sign pairs and project onto a linearly constrained quarter-box.

    Convex dual values give lower bounds for every enumerated subproblem.
    Floating-point values and residuals are diagnostics, not interval certificates.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError('A two-dimensional matrix with at least one column is required.')
    k = matrix.shape[1]
    pattern_count = 2 ** (k - 1)
    if pattern_count > max_patterns:
        raise ValueError('Exhaustive verification exceeds max_patterns.')
    u = cp.Variable(k)
    target = cp.Parameter(k)
    equality = matrix @ u == 0 if len(matrix) else None
    constraints = [u <= .25, -u <= .25]
    if equality is not None:
        constraints.append(equality)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(u - target)), constraints)
    best_upper, best_lower, best_u = np.inf, np.inf, None
    max_equality = max_box = max_stationarity = max_gap = 0.
    started = time.perf_counter()
    for tail in product((-1., 1.), repeat=k - 1):
        sigma = np.r_[1., tail]
        target.value = sigma / 2
        problem.solve(solver='CLARABEL', tol_gap_abs=1e-10,
                      tol_gap_rel=1e-10, tol_feas=1e-10)
        if problem.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError(problem.status)
        value = np.asarray(u.value)
        upper, lower = np.asarray(constraints[0].dual_value), np.asarray(constraints[1].dual_value)
        multiplier = np.asarray(equality.dual_value) if equality is not None else np.empty(0)
        linear = upper - lower + (matrix.T @ multiplier if len(matrix) else 0)
        # Lagrangian infimum for min ||u-target||^2, with exact box RHS 1/4.
        dual = linear @ target.value - linear @ linear / 4 - .25 * np.sum(upper + lower)
        sub_upper = float(np.sum((value-target.value)**2))
        max_gap = max(max_gap, sub_upper-dual)
        max_stationarity = max(max_stationarity, float(np.linalg.norm(2*(value-target.value)+linear, np.inf)))
        max_equality = max(max_equality, float(np.max(np.abs(matrix @ value), initial=0)))
        max_box = max(max_box, float(np.max(np.maximum(np.abs(value)-.25, 0))))
        # 2k - k/2 + 2 distance^2 = 3k/2 + 2 distance^2.
        risk_upper = 1.5*k + 2*sub_upper
        risk_lower = 1.5*k + 2*float(dual)
        best_lower = min(best_lower, risk_lower)
        if risk_upper < best_upper:
            best_upper, best_u = risk_upper, value.copy()
    return dict(k=k, action_dimension=4*k, mixed_risk=float(k),
                baseline=13*k/8, proper_lower=best_lower, proper_upper=best_upper,
                central_risk=central_risk(best_u), center_parameter=best_u.tolist(),
                patterns=pattern_count, seconds=time.perf_counter()-started,
                max_equality_residual=max_equality, max_box_violation=max_box,
                max_stationarity_residual=max_stationarity, max_subproblem_gap=max_gap)
