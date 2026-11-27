"""Run the balanced-capacity multidimensional, global-reference and complete-cost study."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.scalar_capacity import capacity_instance
from src.multidimensional_certification import (
    Projector, all_regions_reference, query_certificate, structured_certificate,
    sampled_ball, distance_checks)


def instance(q, p, seed, capacity_factor, singular_value):
    A, b, c, _, _, _ = capacity_instance(q, seed, capacity_factor)
    rng = np.random.default_rng(seed+217301)
    G = singular_value*np.linalg.qr(rng.normal(size=(q, q)))[0][:, :p]
    singular_values = np.linalg.svd(G, compute_uv=False)
    assert np.linalg.matrix_rank(G) == p
    assert np.max(singular_values)-np.min(singular_values) < 1e-12
    # Under the source box [0,.3]^q this entire box gives the same upper corner.
    observed = np.full(q, .3)
    source_c = observed+np.sum(np.abs(G), axis=1)
    source_margin = float(np.min(source_c-np.sum(np.abs(G), axis=1)-observed))
    assert source_margin >= -1e-12
    return A, b, c, G, dict(minimum_singular_value=float(singular_values.min()),
        maximum_singular_value=float(singular_values.max()), source_margin=source_margin)


def run(output, pilot=False):
    directory = ROOT/'results'/output
    directory.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT/'notes/balanced_capacity_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    started = time.perf_counter()
    rows, references, failures, instances = [], {}, [], {}
    pairs = protocol['reference_pairs']+protocol['scaling_pairs']
    seeds, factors = protocol['seeds'], protocol['capacity_factors']
    budgets, repetitions = protocol['query_budgets'], protocol['timing_repetitions']
    if pilot:
        pairs, seeds, factors, budgets, repetitions = [[4, 2], [16, 4], [32, 8]], seeds[:1], [0.7], [33, 129], 1
    def retain(row):
        rows.append(row)
        with (directory/'cases.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(row)+'\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for q, p in pairs:
            for seed in seeds:
                for factor in factors:
                    identifier = f'q{q}_p{p}_s{seed}_c{factor}'
                    print('Starting '+identifier, flush=True)
                    try:
                        A, b, c, G, design = instance(q, p, seed, factor, protocol['latent_singular_value'])
                        for name, value in [('A', A), ('b', b), ('c', c), ('G', G)]:
                            instances[identifier+'_'+name] = value
                        reference = None
                        if [q, p] in protocol['reference_pairs']:
                            reference = all_regions_reference(A, b, c, G)
                            rng = np.random.default_rng(seed+381)
                            projector = Projector(A, b, q)
                            maximum_error = 0.
                            for v in rng.uniform(-1., 1., size=(protocol['reference_validation_probes'], p)):
                                actual, _ = projector(c+G @ v)
                                cells = [cell for cell in reference['region_data'] if np.max(cell['D'] @ v-cell['e']) < 1e-7]
                                if not cells:
                                    raise RuntimeError('Reference regions do not cover an independent probe')
                                maximum_error = max(maximum_error, min(np.linalg.norm(actual-cell['g']-cell['F'] @ v) for cell in cells))
                            if maximum_error > 2e-6:
                                raise RuntimeError('Global reference failed independent forward probes')
                            references[identifier] = {k:v for k, v in reference.items() if k not in ['region_data','image_vertices']}
                            references[identifier]['probe_error'] = maximum_error
                            references[identifier]['penalty_lower'] = max(0., reference['P_lower']-reference['R_upper'])
                            references[identifier]['penalty_upper'] = max(0., reference['P_upper']-reference['R_lower'])
                            calibration_ball = sampled_ball(reference['image_vertices'])
                            calibration = distance_checks(G, c, A, b, calibration_ball['anchor'])
                            references[identifier]['supplied_center_original_gap'] = max(0.,
                                calibration_ball['variance']+calibration['original_distance']**2-calibration_ball['sample_upper'])
                            references[identifier]['supplied_center_adaptive_gap'] = max(0.,
                                calibration_ball['variance']+calibration['adaptive_distance']**2-calibration_ball['sample_upper'])
                        for budget in budgets:
                            for repetition in range(repetitions):
                                row = query_certificate(A, b, c, G, budget)
                                row.update(kind='capacity', instance=identifier, q=q, p=p, seed=seed,
                                    capacity_factor=factor, budget=budget, repetition=repetition, **design)
                                if reference is not None:
                                    exact_upper = float(np.max(np.sum((reference['image_vertices']-row['anchor'])**2, axis=1)))
                                    row['reference_anchor_upper'] = exact_upper
                                    if row['upper'] < exact_upper-protocol['bound_check_tolerance']:
                                        raise RuntimeError('Global upper bound is below the full reference image')
                                    if row['adaptive_gap'] > references[identifier]['penalty_upper']+protocol['bound_check_tolerance']:
                                        raise RuntimeError('Lower certificate exceeds the global reference penalty')
                                    row['reference_penalty_lower'] = references[identifier]['penalty_lower']
                                    row['reference_penalty_upper'] = references[identifier]['penalty_upper']
                                    row['reference_seconds'] = reference['seconds']
                                retain(row)
                            print(f'  budget {budget} complete; bound={row["adaptive_gap"]:.6g}, '
                                  f'total={row["adaptive_total_seconds"]:.4g}s', flush=True)
                    except Exception as exc:
                        failures.append(dict(instance=identifier, error=repr(exc)))
                        print('FAILED '+identifier+': '+repr(exc), flush=True)
        dimensions = protocol['structured_parameter_dimensions'] if not pilot else [2, 8]
        rotation_seeds = protocol['structured_rotation_seeds'] if not pilot else protocol['structured_rotation_seeds'][:1]
        for p in dimensions:
            for seed in rotation_seeds:
                for affine in protocol['structured_affine_controls']:
                    identifier = f'fold_p{p}_s{seed}_affine{affine}'
                    try:
                        for repetition in range(repetitions):
                            row = structured_certificate(p, seed, affine)
                            row.update(kind='structured', instance=identifier, repetition=repetition)
                            if row['adaptive_gap'] > row['penalty_reference']+protocol['bound_check_tolerance']:
                                raise RuntimeError('Structured certificate exceeds its exact reference')
                            if row['upper'] < row['R_reference']-protocol['bound_check_tolerance']:
                                raise RuntimeError('Structured upper value is below the exact mixed risk')
                            retain(row)
                        print(f'{identifier}: bound={row["adaptive_gap"]:.6g}', flush=True)
                    except Exception as exc:
                        failures.append(dict(instance=identifier, error=repr(exc)))
                        print('FAILED '+identifier+': '+repr(exc), flush=True)
    np.savez_compressed(directory/'instances.npz', **instances)
    report = dict(protocol=protocol, pilot=pilot,
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        rows=rows, references=references, failures=failures,
        seconds=time.perf_counter()-started, warning_count=len(caught),
        warnings=sorted({str(w.message) for w in caught}),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
            processor=os.environ.get('PROCESSOR_IDENTIFIER', platform.processor()),
            logical_cpus=os.cpu_count(), threads=1,
            packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','cvxpy','clarabel']}),
        scope='Balanced full-rank boxes with coupled constraints, independent complete-region small references, '
              'and rotated full-dimensional ball controls. No global small reference is supplied to the query pipeline. '
              'Costs include query setup, queries, all sampled-anchor solves, upper bounds, SOCP checks and bookkeeping. '
              'Primal/dual residuals and numerical backoffs are diagnostics, not interval certification.')
    (directory/'runs.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary = dict(rows=len(rows), capacity_rows=sum(r['kind']=='capacity' for r in rows),
        structured_rows=sum(r['kind']=='structured' for r in rows), references=len(references),
        failures=failures, seconds=report['seconds'], warning_count=len(caught))
    (directory/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise RuntimeError('Incomplete study: all failures retained in runs.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if not args.output or Path(args.output).name != args.output or any(c in args.output for c in '/\\:'):
        raise ValueError('Use a fresh simple output name')
    run(args.output, args.pilot)
