"""Primal/dual normal-cone separation for a finite-dimensional box prior."""
import cvxpy as cp
import numpy as np


def box_cone_distance(latent, offset, normals, anchor, lower, upper):
    """Distance from c+G[lower,upper] to x+cone(normals.T).

    Lower values include a numerical backoff and a checked dual residual;
    they are floating-point diagnostics, not interval-certified bounds.
    """
    G = np.asarray(latent, dtype=float)
    c, x = np.asarray(offset, float), np.asarray(anchor, float)
    q, p = G.shape
    A = np.asarray(normals, float).reshape((-1, q))
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != (p,) or hi.shape != (p,) or not np.isfinite(np.r_[lo, hi]).all() or (lo > hi).any():
        raise ValueError('A finite nonempty parameter box is required')
    v = cp.Variable(p)
    normal = cp.Constant(np.zeros(q))
    if len(A):
        normal = A.T @ cp.Variable(len(A), nonneg=True)
    primal = cp.Problem(cp.Minimize(cp.norm(c + G @ v - x - normal)), [v >= lo, v <= hi])
    settings = dict(solver='CLARABEL', tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=400)
    primal.solve(**settings)
    w = cp.Variable(q)
    coefficients = G.T @ w
    support = cp.sum(cp.maximum(cp.multiply(lo, coefficients), cp.multiply(hi, coefficients)))
    constraints = [cp.norm(w) <= 1]
    if len(A):
        constraints.append(A @ w >= 0)
    dual = cp.Problem(cp.Maximize((x - c) @ w - support), constraints)
    dual.solve(**settings)
    if primal.status not in ['optimal', 'optimal_inaccurate'] or dual.status not in ['optimal', 'optimal_inaccurate']:
        raise RuntimeError('Box separation failed: '+str((primal.status, dual.status)))
    W = np.asarray(w.value).ravel()
    violation = max(0., np.linalg.norm(W)-1., float(np.max(-A @ W)) if len(A) else 0.)
    if violation > 1e-7:
        raise RuntimeError('Box-separation dual is infeasible beyond tolerance')
    point=np.asarray(v.value).ravel()
    primal_violation=max(0.,float(np.max(lo-point)),float(np.max(point-hi)))
    if primal_violation>1e-7 or float(dual.value)>float(primal.value)+1e-6:
        raise RuntimeError('Box separation failed its primal/dual diagnostic')
    return dict(upper=float(primal.value), lower=max(0., float(dual.value)-1e-7),
                primal_status=primal.status, dual_status=dual.status,
                dual_feasibility_residual=float(violation), primal_box_violation=primal_violation,
                primal_dual_gap=float(primal.value-dual.value), numerical_backoff=1e-7)
