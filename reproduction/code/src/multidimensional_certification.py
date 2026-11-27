"""Forward-query certificates and independent small-dimensional references.

All guarantees refer to exact arithmetic. Solver residuals, numerical backoffs,
and complete timing stages are returned; these are not interval computations.
"""
from dataclasses import dataclass
from itertools import combinations, product
import heapq
import time

import clarabel
import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize
from scipy.spatial import HalfspaceIntersection

from .adaptive_certificate import cone_distance, radius_search
from .box_certificate import box_cone_distance


SETTINGS = dict(solver='CLARABEL', tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                tol_feas=1e-10, max_iter=400)


class Projector:
    """Reuse the numerical factorization structure of min ||y-z||^2/2, Ay<=b."""
    def __init__(self, rows, bounds, dimension):
        self.A = np.asarray(rows, float).reshape((-1, dimension))
        self.b = np.asarray(bounds, float)
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
        settings.max_iter = 400
        self.solver = clarabel.DefaultSolver(
            sparse.eye(dimension, format='csc'), np.zeros(dimension),
            sparse.csc_matrix(self.A), self.b,
            [clarabel.NonnegativeConeT(len(self.b))], settings)
        self.calls = 0
        self.maximum_kkt_residual = 0.

    def __call__(self, preferred):
        z = np.asarray(preferred, float)
        self.solver.update(q=-z)
        result = self.solver.solve()
        if str(result.status) not in ('Solved', 'AlmostSolved'):
            raise RuntimeError('Forward projection: ' + str(result.status))
        y, multipliers = np.asarray(result.x), np.asarray(result.z)
        # Interior-point iterates can be O(sqrt(tolerance)) away from a weakly
        # active face. Accept a polished point only after checking both KKT signs.
        if np.max(self.A @ z-self.b, initial=0.) <= 0.:
            y, multipliers = z.copy(), np.zeros(len(self.b))
        else:
            slack = self.b-self.A @ y
            norms = np.linalg.norm(self.A, axis=1)
            candidates = [np.flatnonzero(multipliers*norms >= slack/np.maximum(norms, 1e-30))]
            candidates += [np.flatnonzero(slack < threshold) for threshold in (1e-4, 1e-5, 1e-6, 1e-7)]
            for active in candidates:
                rows = self.A[active]
                if not len(active) or np.linalg.matrix_rank(rows, tol=1e-9) != len(active):
                    continue
                lam = np.linalg.solve(rows @ rows.T, rows @ z-self.b[active])
                candidate = z-rows.T @ lam
                if np.min(lam) >= -1e-11 and np.max(self.A @ candidate-self.b, initial=0.) <= 1e-11:
                    y, multipliers = candidate, np.zeros(len(self.b))
                    multipliers[active] = lam
                    break
        residual = max(0., float(np.max(self.A @ y-self.b, initial=0.)),
                       float(np.max(-multipliers, initial=0.)),
                       float(np.linalg.norm(y-z+self.A.T @ multipliers, np.inf)),
                       float(np.max(np.abs(multipliers*(self.b-self.A @ y)), initial=0.)))
        self.maximum_kkt_residual = max(self.maximum_kkt_residual, residual)
        self.calls += 1
        if residual > 2e-7:
            raise RuntimeError('Forward KKT residual exceeds tolerance')
        return y, multipliers


def affine_region(A, b, c, G, active):
    """An independent active basis gives y=offset+matrix*v and lambda=l0+Lv."""
    q, p = G.shape
    active = np.asarray(active, dtype=int)
    if not len(active):
        return c.copy(), G.copy(), np.empty(0), np.empty((0, p))
    rows = A[active]
    if np.linalg.matrix_rank(rows, tol=1e-9) != len(rows):
        return None
    gram = rows @ rows.T
    l0 = np.linalg.solve(gram, rows @ c-b[active])
    L = np.linalg.solve(gram, rows @ G)
    return c-rows.T @ l0, G-rows.T @ L, l0, L


def sampled_ball(points, tolerance=2e-7):
    """Finite exchange with a feasible contact distribution, not an oracle center."""
    points = np.asarray(points, float)
    chosen = [0]
    farthest = int(np.argmax(np.sum((points-points[0])**2, axis=1)))
    if farthest:
        chosen.append(farthest)
    scale = max(1., float(np.max(np.sum(points*points, axis=1))))
    for iteration in range(len(points)+1):
        selected = points[chosen]
        x, t = cp.Variable(points.shape[1]), cp.Variable()
        constraints = [cp.sum_squares(x-y) <= t for y in selected]
        problem = cp.Problem(cp.Minimize(t), constraints)
        problem.solve(**SETTINGS)
        if problem.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError('Sample enclosing ball: ' + str(problem.status))
        weights = np.maximum([float(np.asarray(c.dual_value).item()) for c in constraints], 0.)
        weights /= weights.sum()
        squared_norms = np.sum(selected*selected, axis=1)
        dual = minimize(lambda w: float(np.sum((w @ selected)**2)-w @ squared_norms),
            weights, jac=lambda w: 2*selected @ (w @ selected)-squared_norms,
            bounds=[(0., 1.)]*len(weights),
            constraints=[{'type':'eq', 'fun':lambda w: w.sum()-1.,
                          'jac':lambda w: np.ones_like(w)}],
            method='SLSQP', options={'ftol':1e-13, 'maxiter':500})
        weights = np.maximum(dual.x, 0.)
        weights /= weights.sum()
        positive = np.flatnonzero(weights > 1e-8)
        support = selected[positive]
        kkt = np.block([[2*support @ support.T, np.ones((len(positive), 1))],
                        [np.ones((1, len(positive))), np.zeros((1, 1))]])
        solution = np.linalg.lstsq(kkt, np.r_[squared_norms[positive], 1.], rcond=None)[0][:-1]
        if np.min(solution) >= -1e-10:
            polished = np.zeros(len(weights))
            polished[positive] = np.maximum(solution, 0.)
            polished /= polished.sum()
            old_variance = weights @ squared_norms-np.sum((weights @ selected)**2)
            new_variance = polished @ squared_norms-np.sum((polished @ selected)**2)
            if new_variance >= old_variance-1e-14:
                weights = polished
        anchor = weights @ selected
        variance = float(weights @ np.sum((selected-anchor)**2, axis=1))
        distances = np.sum((points-anchor)**2, axis=1)
        index = int(np.argmax(distances))
        upper = float(distances[index])
        if upper-variance <= tolerance*scale:
            return dict(anchor=anchor, variance=max(0., variance-1e-9*scale),
                        raw_variance=variance, sample_upper=upper+1e-9*scale,
                        weights=weights, contact_indices=np.array(chosen),
                        exchange_solves=iteration+1, numerical_backoff=1e-9*scale)
        if index in chosen:
            raise RuntimeError('Enclosing ball failed its primal/dual gap check')
        chosen.append(index)
    raise RuntimeError('Finite exchange did not terminate')


def distance_checks(G, c, A, b, anchor, prior='box'):
    """Original and adaptive checks receive exactly the same data and anchor."""
    norms = np.linalg.norm(A, axis=1)
    slack = b-A @ anchor
    active = slack < 2e-7
    inactive = (~active) & (norms > 0)
    margin = float(np.min(slack[inactive]/norms[inactive])) if inactive.any() else np.inf
    diagnostics = []
    def oracle(normals_):
        if prior == 'box':
            result = box_cone_distance(G, c, normals_, anchor,
                                       -np.ones(G.shape[1]), np.ones(G.shape[1]))
        else:
            result = cone_distance(G, c, normals_, anchor, radius=1.)
        diagnostics.append(result)
        return result
    started = time.perf_counter()
    original = oracle(A[active])
    original_seconds = time.perf_counter()-started
    started = time.perf_counter()
    adaptive = radius_search(A, b, anchor, oracle)
    adaptive_seconds = time.perf_counter()-started
    return dict(original_distance=min(margin, original['lower']),
                adaptive_distance=adaptive['lower'],
                original_seconds=original_seconds, adaptive_seconds=adaptive_seconds,
                adaptive_evaluations=adaptive['solves'],
                maximum_dual_residual=max(d['dual_feasibility_residual'] for d in diagnostics),
                maximum_distance_gap=max(d['upper']-d['lower'] for d in diagnostics))


@dataclass
class Leaf:
    center: np.ndarray
    half: np.ndarray
    response: np.ndarray
    affine: object


def query_certificate(A, b, c, G, budget, *, tighten=True):
    """Axis witnesses followed by box subdivision; no reference partition is read.

    Every leaf retains its center query. Nonexpansiveness gives a global cover
    bound. A whole leaf inside a verified affine region admits a vertex maximum.
    The partition is a partition of the parameter BOX, not a reconstruction of
    all optimizer regions. Unresolved leaves always retain the cover bound.
    """
    started = time.perf_counter()
    p, q = G.shape[1], G.shape[0]
    if budget < 2*p+1:
        raise ValueError('Budget must include the center and all axis witnesses')
    forward_seconds = anchor_seconds = upper_seconds = 0.
    forward_start = time.perf_counter()
    projector = Projector(A, b, q)
    forward_seconds += time.perf_counter()-forward_start
    samples = []
    def query(v):
        nonlocal forward_seconds
        begin = time.perf_counter()
        y, dual = projector(c+G @ v)
        forward_seconds += time.perf_counter()-begin
        samples.append(y)
        active = np.flatnonzero((dual > 1e-8) & (b-A @ y < 1e-6))
        return y, affine_region(A, b, c, G, active)
    y, region = query(np.zeros(p))
    leaves = {0: Leaf(np.zeros(p), np.ones(p), y, region)}
    for j in range(p):
        for sign in (-1., 1.):
            v = np.zeros(p)
            v[j] = sign
            query(v)
    opnorm = float(np.linalg.norm(G, 2))
    signs = np.asarray(list(product((-1., 1.), repeat=p)))
    column_norms = np.linalg.norm(G, axis=0)
    ball_solves = 0
    def anchor_update():
        nonlocal anchor_seconds, ball_solves
        begin = time.perf_counter()
        result = sampled_ball(np.array(samples))
        anchor_seconds += time.perf_counter()-begin
        ball_solves += result['exchange_solves']
        return result
    ball = anchor_update()
    def upper(leaf, anchor):
        radius = opnorm*np.linalg.norm(leaf.half)
        plain = (np.linalg.norm(leaf.response-anchor)+radius)**2
        value, affine_used = plain, False
        if tighten and leaf.affine is not None:
            g, F, l0, L = leaf.affine
            preferred_center = g+F @ leaf.center
            target_residual = A @ preferred_center-b+np.abs(A @ F) @ leaf.half
            multiplier_lower = l0+L @ leaf.center-np.abs(L) @ leaf.half
            if (np.max(target_residual, initial=0.) <= 2e-11 and
                    np.min(multiplier_lower, initial=0.) >= -2e-11):
                vertices = (leaf.center+signs*leaf.half) @ F.T+g
                value = min(plain, float(np.max(np.sum((vertices-anchor)**2, axis=1)))
                            +1e-8*max(1., plain))
                affine_used = True
        return value, plain, affine_used
    def heap_update():
        nonlocal upper_seconds
        begin = time.perf_counter()
        heap = [(-upper(leaf, ball['anchor'])[0], index) for index, leaf in leaves.items()]
        heapq.heapify(heap)
        upper_seconds += time.perf_counter()-begin
        return heap
    heap = heap_update()
    next_index, since_anchor = 1, 0
    while projector.calls+2 <= budget:
        _, index = heapq.heappop(heap)
        leaf = leaves.pop(index)
        j = int(np.argmax(column_norms*leaf.half))
        half = leaf.half.copy()
        half[j] /= 2
        for sign in (-1., 1.):
            center = leaf.center.copy()
            center[j] += sign*half[j]
            y, region = query(center)
            child = Leaf(center, half.copy(), y, region)
            leaves[next_index] = child
            begin = time.perf_counter()
            value = upper(child, ball['anchor'])[0]
            upper_seconds += time.perf_counter()-begin
            heapq.heappush(heap, (-value, next_index))
            next_index += 1
        since_anchor += 2
        if since_anchor >= 32:
            ball = anchor_update()
            heap = heap_update()
            since_anchor = 0
    ball = anchor_update()
    begin = time.perf_counter()
    bounds = [upper(leaf, ball['anchor']) for leaf in leaves.values()]
    U = max(item[0] for item in bounds)
    plain_U = max(item[1] for item in bounds)
    upper_seconds += time.perf_counter()-begin
    common_seconds = time.perf_counter()-started
    checks = distance_checks(G, c, A, b, ball['anchor'])
    nu = ball['variance']
    return dict(queries=projector.calls, leaves=len(leaves),
                affine_leaves=sum(item[2] for item in bounds),
                variance=nu, upper=U, plain_upper=plain_U, anchor=ball['anchor'].tolist(),
                original_gap=max(0., nu+checks['original_distance']**2-U),
                adaptive_gap=max(0., nu+checks['adaptive_distance']**2-U),
                original_cover_gap=max(0., nu+checks['original_distance']**2-plain_U),
                adaptive_cover_gap=max(0., nu+checks['adaptive_distance']**2-plain_U),
                original_total_seconds=common_seconds+checks['original_seconds'],
                adaptive_total_seconds=common_seconds+checks['adaptive_seconds'],
                common_seconds=common_seconds, forward_seconds=forward_seconds,
                anchor_seconds=anchor_seconds, upper_seconds=upper_seconds,
                bookkeeping_seconds=max(0., common_seconds-forward_seconds-anchor_seconds-upper_seconds),
                ball_exchange_solves=ball_solves,
                maximum_forward_residual=projector.maximum_kkt_residual, **checks)


def all_regions_reference(A, b, c, G):
    """Enumerate EVERY independent target active subset, then solve every cell.

    Intended only for small boxes. This global reference neither infers regions
    from the certificate's queries nor receives its anchors or lower bounds.
    """
    started = time.perf_counter()
    q, p = G.shape
    regions, image_vertices = [], []
    tested = 0
    for cardinality in range(q+1):
        for active in combinations(range(len(b)), cardinality):
            tested += 1
            affine = affine_region(A, b, c, G, active)
            if affine is None:
                continue
            g, F, l0, L = affine
            D = np.vstack((A @ F, -L, np.eye(p), -np.eye(p)))
            e = np.r_[b-A @ g, l0, np.ones(2*p)]
            norms = np.linalg.norm(D, axis=1)
            zero = norms < 1e-10
            if np.any(e[zero] < -1e-8):
                continue
            D, e, norms = D[~zero], e[~zero], norms[~zero]
            D, e = D/norms[:, None], e/norms
            interior = linprog(np.r_[np.zeros(p), -1.],
                A_ub=np.column_stack((D, np.ones(len(e)))), b_ub=e,
                bounds=[(None, None)]*p+[(0., None)], method='highs')
            if not interior.success or interior.x[-1] < 1e-8:
                continue
            center = interior.x[:-1]
            if p == 1:
                vertices = np.array([[np.max(e[D[:, 0] < 0]/D[D[:, 0] < 0, 0])],
                                     [np.min(e[D[:, 0] > 0]/D[D[:, 0] > 0, 0])]])
            else:
                vertices = HalfspaceIntersection(np.column_stack((D, -e)), center).intersections
            image = vertices @ F.T+g
            image_vertices.extend(image)
            regions.append(dict(D=D, e=e, g=g, F=F, center=center, vertices=vertices))
    if not regions:
        raise RuntimeError('Reference found no full-dimensional optimizer regions')
    vertices = np.unique(np.round(np.array(image_vertices), 11), axis=0)
    # Rounding changes every listed response by at most sqrt(q)*5e-12.
    ball = sampled_ball(vertices)
    projector = Projector(A, b, q)
    lower_values, upper_values = [], []
    for cell in regions:
        v, t = cp.Variable(p), cp.Variable()
        constraints = [cp.sum_squares(cell['g']+cell['F'] @ v-y) <= t for y in vertices]
        cell_constraint = cell['D'] @ v <= cell['e']
        problem = cp.Problem(cp.Minimize(t), constraints+[cell_constraint])
        problem.solve(**SETTINGS)
        if problem.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError('Reference minimax cell: ' + str(problem.status))
        point = np.clip(np.asarray(v.value), -1., 1.)
        response, _ = projector(c+G @ point)
        upper_values.append(float(np.max(np.sum((vertices-response)**2, axis=1))))
        weights = np.maximum([float(np.asarray(con.dual_value).item()) for con in constraints], 0.)
        weights /= weights.sum()
        multipliers = np.maximum(np.asarray(cell_constraint.dual_value), 0.)
        residuals = cell['g']+cell['F'] @ point-vertices
        lagrangian = float(weights @ np.sum(residuals**2, axis=1)
                           +multipliers @ (cell['D'] @ point-cell['e']))
        gradient = 2*cell['F'].T @ (weights @ residuals)+cell['D'].T @ multipliers
        # Convexity bounds the Lagrangian below on the entire parameter box.
        lower_values.append(lagrangian-gradient @ point-np.linalg.norm(gradient, 1)-1e-8)
    Plo, Phi = float(min(lower_values)), float(min(upper_values))
    if Phi-Plo > 3e-5:
        raise RuntimeError('Global reference gap exceeds tolerance')
    return dict(R_lower=ball['variance'], R_upper=ball['sample_upper'],
                P_lower=Plo, P_upper=Phi, regions=len(regions),
                response_vertices=len(vertices), tested_active_subsets=tested,
                seconds=time.perf_counter()-started,
                region_data=regions, image_vertices=vertices,
                maximum_forward_residual=projector.maximum_kkt_residual)


def structured_certificate(p, rotation_seed, affine_control=False):
    """Full-rank unit-ball ambiguity with a known GLOBAL reference, after rotation."""
    started = time.perf_counter()
    rng = np.random.default_rng(rotation_seed)
    rotation = np.linalg.qr(rng.normal(size=(2*p, 2*p)))[0]
    G = rotation @ np.vstack((np.eye(p), -np.eye(p)))
    A = np.empty((0, 2*p)) if affine_control else -rotation.T
    b, c = np.zeros(len(A)), np.zeros(2*p)
    forward_start = time.perf_counter()
    projector = Projector(A, b, 2*p)
    values = []
    for v in np.vstack((np.zeros((1, p)), np.eye(p), -np.eye(p))):
        response, _ = projector(G @ v)
        values.append(response)
    forward_seconds = time.perf_counter()-forward_start
    anchor_start = time.perf_counter()
    ball = sampled_ball(np.array(values))
    anchor_seconds = time.perf_counter()-anchor_start
    upper_start = time.perf_counter()
    if affine_control:
        U = (np.linalg.norm(ball['anchor'])+np.linalg.norm(G, 2))**2
        R, P = 2., 2.
    else:
        coordinates = rotation.T @ ball['anchor']
        if np.min(coordinates) < -1e-7:
            raise RuntimeError('Folded anchor failed nonnegativity')
        # For any a>=0: ||Phi(u)-a||^2 <= ||a||^2+max(0,1-2 min(a)).
        # A 1e-8 guard covers numerical rotation and projection residuals.
        U = float(coordinates @ coordinates+max(0., 1-2*np.min(coordinates))+1e-8)
        R, P = 1-1/(2*p), 1.
    upper_seconds = time.perf_counter()-upper_start
    common_seconds = time.perf_counter()-started
    checks = distance_checks(G, c, A, b, ball['anchor'], prior='ball')
    return dict(p=p, q=2*p, rotation_seed=rotation_seed, affine_control=affine_control,
        queries=projector.calls, variance=ball['variance'], upper=U, R_reference=R,
        P_reference=P, penalty_reference=P-R,
        original_gap=max(0., ball['variance']+checks['original_distance']**2-U),
        adaptive_gap=max(0., ball['variance']+checks['adaptive_distance']**2-U),
        original_total_seconds=common_seconds+checks['original_seconds'],
        adaptive_total_seconds=common_seconds+checks['adaptive_seconds'],
        common_seconds=common_seconds, forward_seconds=forward_seconds,
        anchor_seconds=anchor_seconds, upper_seconds=upper_seconds,
        bookkeeping_seconds=max(0., common_seconds-forward_seconds-anchor_seconds-upper_seconds),
        maximum_forward_residual=projector.maximum_kkt_residual, **checks)
