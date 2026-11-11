"""Reproduce the coupled-capacity calibration study in a new result directory."""
import argparse,hashlib,importlib.metadata,json,os,platform,sys,time,warnings
from pathlib import Path
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
import cvxpy as cp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.projection_paths import polyhedral_line_path,proper_radius,mixed_radius
from src.adaptive_certificate import adaptive_certificate,original_interval_certificate

def capacity_instance(q,seed,change):
    rng=np.random.default_rng(seed)
    # All source preferences c_source+t*d, t>=0, project to the observed box
    # corner. The target couples allocations through shared resource rows.
    d=rng.normal(size=q);d/=np.linalg.norm(d)
    observed=.3*(d>0)
    c=observed+rng.normal(scale=.25,size=q)
    resource=rng.uniform(.2,1.,size=(max(2,q//2),q));resource/=np.linalg.norm(resource,axis=1)[:,None]
    caps=rng.uniform(.7,1.3,size=q)*change
    b_resource=resource@(rng.uniform(.2,.55,size=q)*change)
    A=np.vstack((-np.eye(q),np.eye(q),resource));b=np.r_[np.zeros(q),caps,b_resource]
    return A,b,c,d,0.,4.

def sample_forward(A,b,c,d,grid):
    x=cp.Variable(len(c));t=cp.Parameter()
    problem=cp.Problem(cp.Minimize(.5*cp.sum_squares(x-c-t*d)),[A@x<=b])
    values=[]
    for value in grid:
        t.value=value
        problem.solve(solver='CLARABEL',tol_gap_abs=1e-11,tol_gap_rel=1e-11,tol_feas=1e-11,max_iter=400)
        if problem.status not in ['optimal','optimal_inaccurate']:raise RuntimeError(problem.status)
        values.append(x.value.copy())
    return np.array(values)

def run(output):
    directory = ROOT / 'results' / output
    directory.mkdir(parents=True, exist_ok=False)
    use_multiscale = True
    protocol_path = ROOT / 'configs/capacity_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    report_path = directory / 'runs.json'
    if report_path.exists():
        raise FileExistsError('Preserve previous holdout results')
    counts = protocol['forward_query_counts']
    largest = max(counts)
    cases = []
    failures = []
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for q in protocol['dimensions']:
            for seed in range(protocol['seeds_per_dimension']):
                for change in protocol['capacity_factors']:
                    key = dict(dimension=q, seed=protocol['seed_base'] + seed, capacity_factor=change)
                    try:
                        A, b, c, d, left, right = capacity_instance(q, key['seed'], change)
                        begin = time.perf_counter()
                        vertices, solves, kkt, gap, pieces = polyhedral_line_path(A, b, c, d, left, right, return_pieces=True)
                        if len(vertices) == 1:
                            P = Rlo = Rhi = 0.0
                            center = vertices[0]
                        else:
                            P, _ = proper_radius(vertices)
                            Rlo, Rhi, center = mixed_radius(vertices)
                        recovery_seconds = time.perf_counter() - begin
                        begin = time.perf_counter()
                        du, dl, margin = original_interval_certificate(A, b, c, d, (left, right), center)
                        certificate_seconds = time.perf_counter() - begin
                        actual = max(0.0, P - Rhi)
                        actual_upper = max(0.0, P - Rlo)
                        full_bound = max(0.0, Rlo + min(dl, margin) ** 2 - Rhi)
                        adaptive = None
                        adaptive_seconds = None
                        begin = time.perf_counter()
                        adaptive = adaptive_certificate(d, c, A, b, center, interval=(left, right))
                        adaptive_seconds = time.perf_counter() - begin
                        adaptive_penalty = max(0.0, Rlo + adaptive['lower'] ** 2 - Rhi)
                        if adaptive_penalty > actual_upper + protocol['bound_violation_tolerance']:
                            raise RuntimeError('Adaptive supplied-center value exceeds reference penalty')
                        if adaptive_penalty + 2e-07 < full_bound:
                            raise RuntimeError('Adaptive lower value weaker than original')
                        if full_bound > actual_upper + protocol['bound_violation_tolerance']:
                            raise RuntimeError('Supplied-center lower value exceeds reference penalty')
                        grid = np.linspace(left, right, largest)
                        begin = time.perf_counter()
                        samples = sample_forward(A, b, c, d, grid)
                        sample_seconds = time.perf_counter() - begin
                        forward_error = 0.0
                        for t, x in zip(grid, samples):
                            matching = [p for p in pieces if p[0] - 1e-08 <= t <= p[1] + 1e-08]
                            assert matching
                            forward_error = max(forward_error, min((np.linalg.norm(p[2] + t * p[3] - x) for p in matching)))
                        if forward_error > 2e-06:
                            raise RuntimeError('Full QP sample differs from inferred path')
                        observed = 0.3 * (d > 0)
                        source_error = float(np.max(np.abs(np.clip(observed + grid[:, None] * d, 0.0, 0.3) - observed)))
                        assert source_error < 1e-14
                        budgets = []
                        for count in counts:
                            step = (largest - 1) // (count - 1)
                            assert step * (count - 1) == largest - 1
                            subset = samples[::step]
                            begin = time.perf_counter()
                            nu, sample_upper, anchor = mixed_radius(subset)
                            upper, lower, alpha = original_interval_certificate(A, b, c, d, (left, right), anchor)
                            cover_error = np.linalg.norm(d) * (right - left) / (2 * (count - 1))
                            U = (np.sqrt(sample_upper) + cover_error) ** 2
                            lower_value = float(nu + min(alpha, lower) ** 2 - U)
                            numerical_bound = max(0.0, lower_value)
                            if numerical_bound > actual_upper + protocol['bound_violation_tolerance']:
                                raise RuntimeError('Finite-query lower value exceeds reference penalty')
                            farthest = float(np.max(np.sum((vertices - anchor) ** 2, axis=1)))
                            if U + 2e-07 < farthest:
                                raise RuntimeError('Lipschitz upper value fails validation')
                            budget = dict(queries=count, variance_lower=nu, global_upper=U, numerical_lower_value=lower_value, positive=numerical_bound > protocol['positive_certificate_threshold'], penalty_fraction=numerical_bound / actual if actual > protocol['positive_penalty_threshold'] else None, center_and_certificate_seconds=time.perf_counter() - begin, validated_farthest_squared=farthest, covering_error=cover_error)
                            adapted = adaptive_certificate(d, c, A, b, anchor, interval=(left, right))
                            adapted_value = float(nu + adapted['lower'] ** 2 - U)
                            if max(0.0, adapted_value) > actual_upper + protocol['bound_violation_tolerance']:
                                raise RuntimeError('Adaptive finite-query value exceeds reference penalty')
                            budget.update(adaptive_lower_value=adapted_value, adaptive_positive=adapted_value > protocol['positive_certificate_threshold'], adaptive_conic_solves=adapted['solves'])
                            budgets.append(budget)
                        cases.append(dict(**key, response_segments=len(vertices) - 1, qp_face_queries=solves, recovery_seconds=recovery_seconds, certificate_seconds=certificate_seconds, sample_seconds=sample_seconds, forward_path_error=forward_error, path_kkt_violation=kkt, partition_gap=gap, source_error=source_error, mixed_lower=Rlo, mixed_upper=Rhi, single_risk=P, ratio=P / Rhi if Rhi > 1e-14 else 1.0, actual_penalty=actual, supplied_center_lower=full_bound, supplied_center_fraction=full_bound / actual if actual > protocol['positive_penalty_threshold'] else None, delta_upper=du, delta_lower=dl, margin=margin, finite_query=budgets, adaptive_lower=adaptive_penalty if adaptive else None, adaptive_fraction=adaptive_penalty / actual if adaptive and actual > protocol['positive_penalty_threshold'] else None, adaptive_seconds=adaptive_seconds, adaptive_solves=adaptive['solves'] if adaptive else None))
                    except Exception as error:
                        failures.append(dict(**key, error=repr(error)))
            print('dimension', q, 'complete; cases', len(cases), 'failures', len(failures), flush=True)
    report = dict(protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(), protocol=protocol, cases=cases, failures=failures, seconds=time.perf_counter() - started, warning_count=len(caught), warnings=sorted(set((str(w.message) for w in caught))), environment=dict(python=platform.python_version(), platform=platform.platform(), packages={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'cvxpy', 'clarabel']}, threads=1), scope='Synthetic numerical study. KKT and primal-dual diagnostics use floating point, not interval validation.')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary = []
    for q in protocol['dimensions']:
        rows = [c for c in cases if c['dimension'] == q]
        positive = [c for c in rows if c['actual_penalty'] > protocol['positive_penalty_threshold']]
        row = dict(dimension=q, cases=len(rows), positive_penalties=len(positive), max_ratio=max((c['ratio'] for c in rows)), median_supplied_center_fraction=float(np.median([c['supplied_center_fraction'] for c in positive])) if positive else None, median_certificate_ms=1000 * float(np.median([c['certificate_seconds'] for c in rows])), median_full_recovery_ms=1000 * float(np.median([c['recovery_seconds'] for c in rows])), finite_query_detections={str(k): sum((b['positive'] for c in rows for b in c['finite_query'] if b['queries'] == k)) for k in counts})
        summary.append(row)
        print(json.dumps(row), flush=True)
        row.update(median_adaptive_fraction=float(np.median([c['adaptive_fraction'] for c in positive])) if positive else None, median_adaptive_ms=1000 * float(np.median([c['adaptive_seconds'] for c in rows])), maximum_adaptive_solves=max((c['adaptive_solves'] for c in rows)), adaptive_finite_query_detections={str(k): sum((b['adaptive_positive'] for c in rows for b in c['finite_query'] if b['queries'] == k)) for k in counts})
        print('adaptive', json.dumps(row), flush=True)
    (directory / 'summary.json').write_text(json.dumps(dict(summary=summary, failures=failures, warning_count=len(caught), seconds=report['seconds']), indent=2), encoding='utf-8')
    metadata=dict(protocol_sha256=report['protocol_sha256'],cases=len(cases),
        failures=failures,seconds=report['seconds'],environment=report['environment'],
        warning_count=report['warning_count'],solver_warnings=report['warnings'],
        scope=report['scope'])
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    expected=len(protocol['dimensions'])*protocol['seeds_per_dimension']*len(protocol['capacity_factors'])
    if failures or len(cases)!=expected:raise RuntimeError('Incomplete capacity study; all failures retained in runs.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not args.output or Path(args.output).name!=args.output or any(c in args.output for c in '/\\:'):
        raise ValueError('Use a simple fresh output name')
    run(args.output)


if __name__=='__main__':main()
