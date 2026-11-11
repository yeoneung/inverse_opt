"""Convex-program checks of projection geometry and information restrictions.

Run via run_numerics.py, or --output with a fresh result directory. The routines report floating-point diagnostics.
"""
from itertools import combinations, product
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
import cvxpy as cp
import numpy as np

SOLVER = dict(solver="CLARABEL", tol_gap_abs=1e-11,
              tol_gap_rel=1e-11, tol_feas=1e-11, max_iter=300)


def enclosing(points, action_vertices=None):
    points = np.asarray(points, dtype=float)
    if action_vertices is None:
        x = cp.Variable(points.shape[1])
        constraints = []
    else:
        vertices = np.asarray(action_vertices, dtype=float)
        weights = cp.Variable(len(vertices), nonneg=True)
        x = vertices.T @ weights
        constraints = [cp.sum(weights) == 1]
    radius = cp.Variable(nonneg=True)
    constraints += [cp.norm(x - y) <= radius for y in points]
    problem = cp.Problem(cp.Minimize(radius), constraints)
    problem.solve(**SOLVER)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(problem.status)
    return float(radius.value**2), np.asarray(x.value)


def facet_checks(configuration):
    rows = []
    for dimension, seed, family in product(configuration["dimensions"], configuration["seeds"], configuration["families"]):
        rng = np.random.default_rng(seed + dimension)
        rotation, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        if family == "box":
            base = np.zeros(dimension); base[-1] = -1
            matrix = np.vstack([np.eye(dimension), -np.eye(dimension)])
            bounds = np.ones(2 * dimension)
        else:
            base = np.r_[np.full(dimension - 1, 1 / (2 * (dimension - 1))), 0.]
            matrix = np.vstack([-np.eye(dimension), np.ones((1, dimension))])
            bounds = np.r_[np.zeros(dimension), 1.]
        matrix = matrix @ rotation.T
        base = rotation @ base
        inward = rotation[:, -1]; tangent = rotation[:, 0]
        slacks = bounds - matrix @ base
        positive = slacks > 1e-8
        margin = float(np.min(slacks[positive] / np.linalg.norm(matrix[positive], axis=1)))
        latent = cp.Parameter(dimension)
        right_hand_side = cp.Parameter(len(bounds))
        action = cp.Variable(dimension)
        projection = cp.Problem(cp.Minimize(cp.sum_squares(action - latent)),
                                [matrix @ action <= right_hand_side])
        for beta, fraction in product(configuration["tangent_energy_fractions"], configuration["relative_radii"]):
            radius = fraction * margin
            right_hand_side.value = slacks / radius
            direction = np.sqrt(beta) * tangent + np.sqrt(1 - beta) * inward
            actual = []
            projection_error = 0.
            for t in np.linspace(-1, 1, 13):
                latent.value = t * direction
                projection.solve(**SOLVER)
                if projection.status != cp.OPTIMAL:
                    raise RuntimeError(projection.status)
                obtained = np.asarray(action.value)
                expected = t * (direction if t >= 0 else np.sqrt(beta) * tangent)
                projection_error = max(projection_error, float(np.linalg.norm(obtained - expected)))
                actual.append(obtained)
            # Endpoints and the kink are taken from a general QP projection,
            # followed by independent unrestricted / two branch SOCP solves.
            left, kink, right = actual[0], actual[6], actual[-1]
            contacts = np.array([left, kink, right])
            mixed, center = enclosing(contacts)
            proper = min(enclosing(contacts, [left, kink])[0],
                         enclosing(contacts, [kink, right])[0])
            # Enlarge the latent segment by its target midpoint. A zero source
            # observation Bv=0 removes that midpoint and recovers the segment.
            # The two response cells of this triangular prior have these vertices.
            exact_center = np.sqrt(1 - beta) * inward / 2
            boundary_crossing = -np.sqrt(beta) * tangent / 3
            positive_cell = np.array([right, exact_center, np.zeros(dimension), boundary_crossing])
            negative_cell = np.array([left, np.zeros(dimension)])
            before_contacts = np.vstack([positive_cell, negative_cell])
            before_mixed, _ = enclosing(before_contacts)
            before_proper = min(enclosing(before_contacts, positive_cell)[0],
                                enclosing(before_contacts, negative_cell)[0])
            expected_mixed = (1 + 3 * beta) / 4
            expected_proper = (1 + 3 * beta)**2 / (4 * (1 + beta)**2)
            rows.append(dict(dimension=dimension, seed=seed, family=family,
                             beta=beta, radius=radius, mixed_normalized=mixed,
                             proper_normalized=proper, ratio=proper / mixed,
                             projection_error=projection_error,
                             risk_error=max(abs(mixed - expected_mixed), abs(proper - expected_proper)),
                             monotonicity_error=max(abs(before_mixed - expected_mixed),
                                                    abs(before_proper - expected_mixed)),
                             source_observation_raises_proper_risk=proper > before_proper,
                             center_error=float(np.linalg.norm(center - (left + right) / 2))))
    return rows


def vertices(matrix, bounds):
    result = []
    for indices in combinations(range(len(bounds)), 2):
        block = matrix[list(indices)]
        if abs(np.linalg.det(block)) < 1e-10:
            continue
        point = np.linalg.solve(block, bounds[list(indices)])
        if np.max(matrix @ point - bounds) <= 1e-9:
            if not any(np.linalg.norm(point - old) < 1e-8 for old in result):
                result.append(point)
    return np.asarray(result)


def strip_risks(upper, width):
    matrix = np.vstack([np.eye(2), -np.eye(2)])
    bounds = np.r_[np.full(2, upper), np.ones(2)]
    if width is not None:
        matrix = np.vstack([matrix, np.ones(2), -np.ones(2)])
        bounds = np.r_[bounds, width, width]
    cells = []
    for signs in product([-1., 1.], repeat=2):
        cell_matrix = np.vstack([matrix, -np.diag(signs)])
        cell_bounds = np.r_[bounds, np.zeros(2)]
        points = vertices(cell_matrix, cell_bounds)
        if len(points):
            cells.append(np.maximum(points, 0.))
    contacts = np.vstack(cells)
    mixed, center = enclosing(contacts)
    proper = min(enclosing(contacts, cell)[0] for cell in cells)
    return mixed, proper, center


def information_checks(configuration):
    rows = []
    for upper in configuration["upper_bounds"]:
        before_r, before_p, _ = strip_risks(upper, None)
        for fraction in configuration["strip_fractions"]:
            width = (upper - 1) * fraction
            mixed, proper, _ = strip_risks(upper, width)
            expected_r = (1 + width)**2 / 2
            expected_p = 1 + width + width**2 / 2
            rows.append(dict(upper=upper, source_tolerance=width,
                             source_loss_budget=width**2 / 2,
                             before_mixed=before_r, before_proper=before_p,
                             after_mixed=mixed, after_proper=proper,
                             error=max(abs(before_r - upper**2 / 2),
                                       abs(before_p - upper**2 / 2),
                                       abs(mixed - expected_r), abs(proper - expected_p))))
    return rows



ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='classification')
    args=parser.parse_args()
    if not args.output or Path(args.output).name!=args.output or any(c in args.output for c in '/\\:'):
        raise ValueError('Use a simple output name')
    output=ROOT/'results'/args.output
    if output.exists():
        raise FileExistsError('Preserve previous results; choose a fresh output name')
    output.mkdir(parents=True)
    protocol=ROOT/'configs/classification_protocol.json'
    configuration=json.loads(protocol.read_text(encoding='utf-8'))
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        facets=facet_checks(configuration)
        information=information_checks(configuration)
    for name,rows in [('facets',facets),('information',information)]:
        with (output/(name+'.csv')).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    analysis=dict(facet_cases=len(facets),information_cases=len(information),
        max_facet_projection_error=max(x['projection_error'] for x in facets),
        max_facet_risk_error=max(x['risk_error'] for x in facets),
        max_monotonicity_error=max(x['monotonicity_error'] for x in facets),
        max_information_error=max(x['error'] for x in information),
        every_facet_has_strict_gap=all(x['proper_normalized']>x['mixed_normalized'] for x in facets),
        every_facet_has_information_reversal=all(x['source_observation_raises_proper_risk'] for x in facets),
        every_information_case_reverses_proper_monotonicity=all(
            x['after_proper']>x['before_proper'] and x['after_mixed']<x['before_mixed'] for x in information),
        scope='Constructed polyhedral QP/SOCP checks; not interval-validated or external replication.')
    assert analysis['facet_cases']==96 and analysis['information_cases']==15
    assert analysis['max_facet_projection_error']<5e-4 and analysis['max_facet_risk_error']<5e-5
    assert analysis['max_monotonicity_error']<5e-5 and analysis['max_information_error']<1e-7
    assert analysis['every_facet_has_strict_gap'] and analysis['every_facet_has_information_reversal']
    assert analysis['every_information_case_reverses_proper_monotonicity']
    analysis['passed']=True
    (output/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    metadata=dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        documentation_status=configuration['documentation_status'],
        python=platform.python_version(),cvxpy=cp.__version__,numpy=np.__version__,
        seconds=time.perf_counter()-started,solver_settings=SOLVER,
        solver_warning_count=len(seen),solver_warnings=[str(w.message) for w in seen])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':main()
