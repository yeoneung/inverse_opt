"""Matched multidimensional inverse classes, covering bounds and separated costs."""
import argparse, hashlib, importlib.metadata, json, os, platform, sys, time, warnings
from pathlib import Path
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name] = '1'
import cvxpy as cp
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.capacity import capacity_instance, sample_forward
from src.projection_paths import polyhedral_line_path, proper_radius, mixed_radius
from src.adaptive_certificate import radius_search
from src.box_certificate import box_cone_distance


def run(output):
    directory = ROOT/'results'/output
    directory.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT/'configs/multidimensional_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    cases, failures, bases = [], [], []
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for q in protocol['action_dimensions']:
            for seed in protocol['seeds']:
                A, b, c, d, left, right = capacity_instance(q, seed, 1.)
                begin = time.perf_counter()
                vertices, _, _, _ = polyhedral_line_path(A, b, c, d, left, right)
                P0, _ = proper_radius(vertices)
                Rlo, Rhi, _ = mixed_radius(vertices)
                line_seconds = time.perf_counter()-begin
                rng = np.random.default_rng(seed+10000)
                matrix = np.column_stack((d, rng.normal(size=(q, q-1))))
                basis = np.linalg.qr(matrix)[0][:, 1:]
                assert np.linalg.norm(basis.T @ d) < 1e-10
                query_records = []
                for count in protocol['query_counts']:
                    grid = np.linspace(left, right, count)
                    begin = time.perf_counter()
                    samples = sample_forward(A, b, c, d, grid)
                    forward_seconds = time.perf_counter()-begin
                    begin = time.perf_counter()
                    nu, sample_upper, anchor = mixed_radius(samples)
                    anchor_seconds = time.perf_counter()-begin
                    query_records.append(dict(queries=count, samples=samples, variance=nu,
                        sample_upper=sample_upper, anchor=anchor, forward_seconds=forward_seconds,
                        anchor_seconds=anchor_seconds))
                bases.append(dict(q=q, seed=seed, P0=P0, R0_lower=Rlo, R0_upper=Rhi,
                    line_recovery_seconds=line_seconds, response_segments=len(vertices)-1))
                for p in protocol['parameter_dimensions']:
                    for eta in protocol['nuisance_radii']:
                        key = dict(q=q, p=p, seed=seed, eta=eta)
                        try:
                            G = np.column_stack((2*d, eta*basis[:, :p-1]/np.sqrt(p-1)))
                            offset = c+2*d
                            assert np.linalg.matrix_rank(G) == p
                            # A source box at its upper corner hides every parameter.
                            observed = np.full(q, .3)
                            source_offset = observed + np.sum(np.abs(G), axis=1)
                            source_violation = float(np.max(np.maximum(.3-source_offset+np.sum(np.abs(G), axis=1), 0)))
                            assert source_violation < 1e-12
                            penalty_lo = max(0., max(0., np.sqrt(P0)-eta)**2-(np.sqrt(Rhi)+eta)**2)
                            penalty_hi = max(0., (np.sqrt(P0)+eta)**2-Rlo)
                            records = []
                            for data in query_records:
                                count, anchor = data['queries'], data['anchor']
                                begin = time.perf_counter()
                                cover = 2/(count-1)+eta
                                U = (np.sqrt(data['sample_upper'])+cover)**2
                                upper_seconds = time.perf_counter()-begin
                                slack = b-A @ anchor
                                active = slack < 2e-7
                                alpha = float(np.min(slack[~active]/np.linalg.norm(A[~active], axis=1))) if (~active).any() else np.inf
                                diagnostics=[]
                                def oracle(normals):
                                    result=box_cone_distance(G, offset, normals, anchor, -np.ones(p), np.ones(p))
                                    diagnostics.append(result)
                                    return result
                                begin = time.perf_counter()
                                original = oracle(A[active])
                                original_seconds = time.perf_counter()-begin
                                begin = time.perf_counter()
                                adaptive = radius_search(A, b, anchor, oracle)
                                adaptive_seconds = time.perf_counter()-begin
                                old_value = max(0., data['variance']+min(alpha, original['lower'])**2-U)
                                new_value = max(0., data['variance']+adaptive['lower']**2-U)
                                assert new_value+3e-7 >= old_value
                                assert new_value <= penalty_hi+protocol['validation_tolerance']
                                # Independent projected responses probe the full box.
                                probe = rng.uniform(-1, 1, size=(protocol['validation_probes'], p))
                                z = offset+probe @ G.T
                                variable = cp.Variable(q)
                                center = cp.Parameter(q)
                                problem = cp.Problem(cp.Minimize(.5*cp.sum_squares(variable-center)), [A @ variable <= b])
                                max_probe = 0.
                                for latent in z:
                                    center.value = latent
                                    problem.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
                                    if problem.status not in ['optimal','optimal_inaccurate']:
                                        raise RuntimeError(problem.status)
                                    max_probe = max(max_probe, float(np.sum((variable.value-anchor)**2)))
                                assert max_probe <= U+protocol['validation_tolerance']
                                records.append(dict(queries=count, variance_lower=data['variance'], global_upper=U,
                                    covering_error=cover, original_lower=old_value, adaptive_lower=new_value,
                                    original_positive=bool(old_value>1e-6), adaptive_positive=bool(new_value>1e-6),
                                    adaptive_distances=adaptive['solves'], candidate_thresholds=adaptive['thresholds'],
                                    forward_seconds=data['forward_seconds'], anchor_seconds=data['anchor_seconds'],
                                    upper_bound_seconds=upper_seconds, original_seconds=original_seconds,
                                    adaptive_seconds=adaptive_seconds,
                                    adaptive_total_seconds=data['forward_seconds']+data['anchor_seconds']+upper_seconds+adaptive_seconds,
                                    validated_probe_max=max_probe,
                                    maximum_dual_residual=max(r['dual_feasibility_residual'] for r in diagnostics),
                                    maximum_primal_box_violation=max(r['primal_box_violation'] for r in diagnostics),
                                    maximum_primal_dual_gap=max(r['primal_dual_gap'] for r in diagnostics)))
                            cases.append(dict(**key, source_error=source_violation, latent_rank=int(np.linalg.matrix_rank(G)),
                                reference_penalty_lower=penalty_lo, reference_penalty_upper=penalty_hi,
                                line_P=P0, line_R_lower=Rlo, line_R_upper=Rhi, measurements=records))
                        except Exception as error:
                            failures.append(dict(**key, error=repr(error)))
                print('q',q,'seed',seed,'conditions',len(cases),'failures',len(failures),flush=True)
    summary = []
    for p in protocol['parameter_dimensions']:
        for eta in protocol['nuisance_radii']:
            selected = [c for c in cases if c['p']==p and c['eta']==eta]
            for count in protocol['query_counts']:
                measured = [next(m for m in c['measurements'] if m['queries']==count) for c in selected]
                summary.append(dict(p=p, eta=eta, queries=count, cases=len(selected),
                    positive_reference_lower=sum(c['reference_penalty_lower']>1e-7 for c in selected),
                    original_detections=sum(m['original_positive'] for m in measured),
                    adaptive_detections=sum(m['adaptive_positive'] for m in measured),
                    median_forward_ms=1000*float(np.median([m['forward_seconds'] for m in measured])),
                    median_anchor_ms=1000*float(np.median([m['anchor_seconds'] for m in measured])),
                    median_upper_ms=1000*float(np.median([m['upper_bound_seconds'] for m in measured])),
                    median_original_ms=1000*float(np.median([m['original_seconds'] for m in measured])),
                    median_adaptive_ms=1000*float(np.median([m['adaptive_seconds'] for m in measured])),
                    median_total_ms=1000*float(np.median([m['adaptive_total_seconds'] for m in measured])),
                    maximum_distances=max(m['adaptive_distances'] for m in measured)))
    report = dict(protocol=protocol, protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        base_contexts=bases, cases=cases, failures=failures, summary=summary,
        seconds=time.perf_counter()-started, warning_count=len(caught),
        warnings=sorted({str(w.message) for w in caught}),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
            packages={name:importlib.metadata.version(name) for name in ['numpy','cvxpy','clarabel']}, threads=1),
        scope='Full-rank anisotropic parameter boxes. Scalar paths plus Hausdorff envelopes validate risk brackets; no full multidimensional partition or global minimax reference is computed. Paired conditions share query and anchor measurements. Floating-point primal/dual diagnostics are not interval certification.')
    def scalar(value):
        if isinstance(value,np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    (directory/'runs.json').write_text(json.dumps(report,indent=2,default=scalar),encoding='utf-8')
    (directory/'summary.json').write_text(json.dumps(dict(summary=summary, failures=failures, seconds=report['seconds'], warning_count=len(caught)),indent=2,default=scalar),encoding='utf-8')
    expected=len(protocol['action_dimensions'])*len(protocol['seeds'])*len(protocol['parameter_dimensions'])*len(protocol['nuisance_radii'])
    if failures or len(cases)!=expected:
        raise RuntimeError('Incomplete multidimensional study; all failures retained')
    print(json.dumps(dict(cases=len(cases),failures=len(failures),seconds=report['seconds']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not args.output or Path(args.output).name!=args.output or any(c in args.output for c in '/\\:'):
        raise ValueError('Use a fresh simple output name')
    run(args.output)
