"""Reproduce the smooth-curve, rational constraint-change and HJ disk controls."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.smooth_curvature import (curvature_coefficient, disk_recovery,
                                 parabolic_recovery, two_endpoint_recovery)


def run(output):
    if not output or Path(output).name != output or any(c in output for c in '/\\:'):
        raise ValueError('Use a simple fresh output directory name.')
    directory = ROOT / 'results' / output
    directory.mkdir(parents=True, exist_ok=False)
    protocol = ROOT / 'configs/curvature_protocol.json'
    settings = json.loads(protocol.read_text())
    started = time.perf_counter()
    curve = lambda t: np.array([t + 0.4 * t**2, 0.7 * t**2 + 0.3 * t**3])
    coefficient = curvature_coefficient([1, 0], [0.8, 1.4])
    smooth = []
    for radius in settings['radii']:
        second_bound = np.linalg.norm([0.8, 1.4]) + 1.8 * radius
        answer = two_endpoint_recovery(curve, radius, 1 - radius * second_bound,
                                       1 + radius * second_bound, second_bound)
        points = np.array([curve(t) for t in np.linspace(-radius, radius, settings['sample_points'])])
        risk = lambda t: float(np.max(np.sum((points - curve(t))**2, axis=1)))
        direct = minimize_scalar(risk, bounds=(-radius, radius), method='bounded',
                                 options={'xatol': 1e-15})
        disagreement = abs(direct.fun - answer['single_risk'])
        assert direct.success and disagreement < settings['minimax_absolute_tolerance']
        sampled_radius = np.max(np.sum((points - answer['mixed_center'])**2, axis=1))
        assert sampled_radius <= answer['mixed_risk'] + 1e-13
        smooth.append(dict(radius=radius, **answer,
                           scaled_penalty=answer['penalty'] / radius**4,
                           predicted_coefficient=coefficient,
                           sample_minimax_disagreement=float(disagreement)))

    parabolic = []
    for value in settings['curvatures']:
        curvature = Fraction(value)
        for value_b in settings['parabolic_endpoints']:
            endpoint = Fraction(value_b)
            radius = endpoint + curvature**2 * endpoint**3 / 2
            mixed = endpoint**2
            gap = curvature**2 * endpoint**4 / 4
            energy = ((radius - endpoint)**2 + gap) / 2
            assert curvature * endpoint <= 2
            assert gap == 2 * energy / (1 + curvature**2 * endpoint**2)
            numeric = parabolic_recovery(float(curvature), float(radius))
            assert abs(numeric['penalty'] - float(gap)) < 1e-12
            parabolic.append(dict(curvature=str(curvature), endpoint=str(endpoint),
                                  radius=str(radius), mixed_risk=str(mixed),
                                  single_risk=str(mixed + gap), penalty=str(gap),
                                  hj_energy=str(energy), energy_identity=True,
                                  scaled_penalty=float(gap / radius**4),
                                  predicted_coefficient=float(curvature**2 / 4)))

    disks = []
    for height in settings['disk_heights']:
        for radius in settings['radii']:
            answer = disk_recovery(height, radius)
            assert abs(answer['single_risk'] - answer['mixed_risk'] - answer['penalty']) < 1e-13
            disks.append(dict(height=height, radius=radius, **answer,
                              scaled_penalty=answer['penalty'] / radius**4))
    paired = []
    for radius in settings['paired_radii']:
        for curvature in settings['paired_curvatures']:
            answer = parabolic_recovery(curvature, radius)
            assert (answer['penalty'] == 0) == (curvature == 0)
            paired.append(dict(radius=radius, curvature=curvature,
                               compatible_interval=[-radius, radius], **answer,
                               scaled_penalty=answer['penalty'] / radius**4,
                               predicted_coefficient=curvature**2 / 4))
    results = dict(smooth_curve=smooth, parabolic_controls=parabolic,
                   disk_controls=disks, fixed_interval_controls=paired)
    summary = dict(smooth_cases=len(smooth), exact_parabolic_cases=len(parabolic),
                   disk_cases=len(disks), fixed_interval_cases=len(paired),
                   total_cases=len(smooth) + len(parabolic) + len(disks) + len(paired),
                   max_sample_minimax_disagreement=max(c['sample_minimax_disagreement'] for c in smooth),
                   smallest_radius=smooth[-1]['radius'],
                   smallest_radius_scaled_penalty=smooth[-1]['scaled_penalty'],
                   predicted_smooth_coefficient=coefficient, rational_energy_identities=True)
    (directory / 'runs.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    metadata = dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                    seconds=time.perf_counter() - started,
                    scope='Local scalar-ambiguity controls; sampled minimax checks are numerical diagnostics.')
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='mor_curvature_final')
    run(parser.parse_args().output)
