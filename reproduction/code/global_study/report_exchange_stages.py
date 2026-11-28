"""Separate corner initialization from subsequent witness updates in saved runs.

This deterministic reanalysis uses recorded solver calls and independent
anchor checks. It performs no new optimization and makes no matched-budget
claim about the partitioning benchmark.
"""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent
TOLERANCE = 1e-6


def build_report(study_root=ROOT):
    root = Path(study_root)
    names = ['capacity_p4_validation.json',
             'compact_capacity_extended/results.json',
             'independent_result_checks.json']
    loaded = {name: json.loads((root / name).read_text(encoding='utf-8'))
              for name in names}
    p4 = loaded[names[0]]['cases']
    p8 = loaded[names[1]]['cases']
    checked = {row['label']: row for row in loaded[names[2]]['rows']}
    groups = {}
    definitions = [
        ('capacity_q16_p4', [row for row in p4 if row['q'] == 16]),
        ('capacity_q32_p4', [row for row in p4 if row['q'] == 32]),
        ('capacity_q12_p8', p8),
    ]
    for key, source_rows in definitions:
        rows = []
        for source in source_rows:
            history = source['history']
            audit = checked[source['label']]
            assert history and [h['iteration'] for h in history] == list(range(len(history)))
            assert len(audit['anchors']) == len(history)
            assert source['initial_queries'] == 2 ** source['p']
            first = history[0]['certificate'] > TOLERANCE
            first_check = audit['anchors'][0]['recomputed_certificate'] > TOLERANCE
            assert first == first_check
            if first:
                # These first-call positives are also final anchors and have
                # undergone complete witness reconstruction/upper transfer.
                assert len(history) == 1 and audit['detected']
                assert audit['validated_certificate'] > TOLERANCE
            rows.append({
                'label': source['label'],
                'initial_corner_queries': source['initial_queries'],
                'global_solver_calls': len(history),
                'post_corner_updates_used_for_last_anchor': len(history) - 1,
                'first_call_certificate': history[0]['certificate'],
                'first_call_independently_rechecked_certificate': audit['anchors'][0]['recomputed_certificate'],
                'detected_on_first_call': first,
                'final_detected': audit['detected'],
                'final_validated_certificate': audit['validated_certificate'],
                'stopping_reason': source['stopping_reason'],
            })
        calls = [row['global_solver_calls'] for row in rows]
        groups[key] = {
            'q': source_rows[0]['q'], 'p': source_rows[0]['p'],
            'cases': len(rows),
            'first_call_detections': sum(row['detected_on_first_call'] for row in rows),
            'final_detections': sum(row['final_detected'] for row in rows),
            'detections_after_added_responses': sum(row['final_detected'] and not row['detected_on_first_call'] for row in rows),
            'global_solver_calls_median': statistics.median(calls),
            'global_solver_calls_minimum': min(calls),
            'global_solver_calls_maximum': max(calls),
            'global_solver_calls_total': sum(calls),
            'rows': rows,
        }
    assert [group['first_call_detections'] for group in groups.values()] == [2, 6, 0]
    assert [group['final_detections'] for group in groups.values()] == [6, 6, 5]
    return {
        'passed': True,
        'scope': 'Deterministic saved-run analysis; no optimization rerun or newly timed ablation.',
        'detection_tolerance': TOLERANCE,
        'first_call_definition': 'All 2**p parameter-box corner responses, one sampled anchor/separation calculation and one global risk solve, before any newly found response is added.',
        'call_count_definition': 'Number of recorded history entries; includes the final successful or inconclusive global call.',
        'update_count_definition': 'Number of response additions used by the last recorded anchor, equal to solver calls minus one. No claim is made about an unused addition after a final budget stop.',
        'validation_scope': 'Every first-call classification agrees with the saved independent anchor check. All eight first-call positives are final one-call runs and also pass full witness reconstruction and conservative upper-risk transfer.',
        'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in names},
        'groups': groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Fresh JSON output path; otherwise print only.')
    args = parser.parse_args()
    text = json.dumps(build_report(), indent=2)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
