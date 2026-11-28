"""Check the saved global-solver/application evidence without optional solvers.

This validates data integrity and recorded arithmetic, not solver proofs.
Use global_study/validate_research_results.py for a new independent QP/SOCP audit.
"""
from pathlib import Path
import hashlib
import json
import math
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'code' / 'global_study'


def load(name):
    return json.loads((STUDY / name).read_text(encoding='utf-8'))


def close(actual, expected, atol=1e-11):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=atol), (actual, expected)


def main():
    manifest = load('artifact_manifest.json')
    for entry in manifest['files']:
        path = STUDY / entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['portable_sha256'], path
        if entry['source_bytes_preserved']:
            assert entry['source_sha256'] == entry['portable_sha256']

    p4 = load('capacity_p4_validation.json')
    p8 = load('compact_capacity_extended/results.json')
    battery = load('battery_grid_validation/runs.json')
    assert p4['complete'] and len(p4['cases']) == 12
    assert p8['complete'] and len(p8['cases']) == 6 and not p8['failures']
    assert len(battery['rows']) == 9 and all(row['passed'] for row in battery['rows'])
    raw = {row['label']: row for row in p4['cases'] + p8['cases']}
    raw.update({row['exchange']['label']: row['exchange'] for row in battery['rows']})
    assert len(raw) == 27
    for row in p4['cases']:
        assert row['baseline']['queries'] == 513
        assert row['baseline']['adaptive_gap'] == 0
    baseline = load('compact_capacity_extended/baseline_query_results.json')
    assert baseline['complete'] and len(baseline['cases']) == 6
    assert all(row['queries'] == 513 and row['adaptive_gap'] == 0 for row in baseline['cases'])
    for row in p8['cases']:
        path = STUDY / 'compact_capacity_extended' / row['instance_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['instance_sha256']
        with np.load(path) as arrays:
            assert arrays['G'].shape == (12, 8)
            assert np.linalg.matrix_rank(arrays['G']) == 8
            assert np.max(np.abs(np.linalg.svd(arrays['G'], compute_uv=False) - 0.75)) < 1e-12
    pilot = load('compact_capacity_pilot/results.json')
    assert len(pilot['cases']) == 1 and not pilot['cases'][0]['detected']
    assert pilot['protocol']['maximum_exchanges'] == 6
    assert p8['protocol']['maximum_exchanges'] == 20

    fixed = load('fixed_anchor_corner_diagnostic.json')
    assert fixed['cases'] == 18 and fixed['all_fixed_anchors_obstructed']
    for row in fixed['rows']:
        close(row['positive_certificate_threshold'], row['variance'] + row['adaptive_squared_separation'])
        assert row['corner_incumbent_lower'] > row['positive_certificate_threshold']
        assert not row['fixed_anchor_rescuable']

    checked = load('independent_result_checks.json')
    assert checked['passed'] and checked['cases'] == len(checked['rows']) == 27
    assert checked['anchors'] == sum(len(row['anchors']) for row in checked['rows']) == 92
    for row in checked['rows']:
        old = raw[row['label']]['history'][-1]
        expected_upper = (math.sqrt(old['oracle']['upper']) + row['anchor_shift']) ** 2
        close(row['transferred_global_upper'], expected_upper)
        expected = max(0, row['validated_variance'] + row['validated_distance'] ** 2 - expected_upper)
        close(row['validated_certificate'], expected)
        assert row['detected'] == (expected > 1e-6)
        assert row['maximum_independent_projection_kkt_residual'] < 1e-8
        close(row['pipeline_and_repair_seconds'], row['original_pipeline_seconds'] + row['independent_repair_seconds'])

    summary = load('validated_summary.json')
    groups = {
        'capacity_q16_p4': [row for row in checked['rows'] if row['label'].startswith('p4validation_q16_')],
        'capacity_q32_p4': [row for row in checked['rows'] if row['label'].startswith('p4validation_q32_')],
        'capacity_q12_p8': [row for row in checked['rows'] if row['label'].startswith('compact_capacity_extended/')],
        'battery_q8_p4': [row for row in checked['rows'] if row['label'].startswith('battery_grid_')],
    }
    assert [sum(row['detected'] for row in rows) for rows in groups.values()] == [6, 6, 5, 7]
    for key, rows in groups.items():
        recorded = summary[key]
        assert recorded['cases'] == len(rows)
        assert recorded['positive'] == sum(row['detected'] for row in rows)
        times = [row['pipeline_and_repair_seconds'] for row in rows]
        close(recorded['median_total_seconds_including_final_validation'], statistics.median(times))
        close(recorded['maximum_total_seconds_including_final_validation'], max(times))
        close(recorded['minimum_positive_bound'], min(row['validated_certificate'] for row in rows if row['detected']))

    reference = load('battery_tight_reference.json')
    assert reference['passed'] and len(reference['rows']) == 2
    primary = reference['rows'][0]
    close(primary['actual_penalty_lower'], primary['proper_risk_lower'] - primary['mixed_risk_upper'])
    close(primary['actual_penalty_upper'], primary['proper_risk_upper'] - primary['mixed_risk_lower'])
    close(primary['squared_risk_ratio_interval'][0], primary['proper_risk_lower'] / primary['mixed_risk_upper'])
    close(primary['squared_risk_ratio_interval'][1], primary['proper_risk_upper'] / primary['mixed_risk_lower'])
    for key, value in summary['primary_battery_reference'].items():
        if isinstance(value, list):
            assert value == primary[key]
        else:
            close(value, primary[key])
    assert primary['actual_penalty_lower'] > 0
    assert load('global_upper_self_checks.json')['passed']
    gurobi = load('gurobi_selfchecks.json')
    assert gurobi['passed'] and gurobi['global_solves'] == 24
    assert gurobi['all_bounds_bracket_independent_complete_region_values']
    assert load('capacity_dual_bounds_checks.json')['passed']
    unresolved = load('coupled_cut_pilot.json')
    assert unresolved['certificate'] == 0
    print(json.dumps({'passed': True, 'cases': 27, 'anchor_checks': 92,
                      'positive_counts': [6, 6, 5, 7],
                      'raw_artifacts_hashed': len(manifest['files']),
                      'scope': 'Saved data, dimensions, bound arithmetic and integrity; no global-solver rerun.'}, indent=2))


if __name__ == '__main__':
    main()
