# Inverse optimization numerics

Numerical code for quadratic projections, minimax recovery, convex separation
certificates, and smooth-response curvature.

Use Python 3.12:

```sh
python -m pip install -r requirements.txt
python -I run_checks.py
python -I reproduction/verification/check_balanced_results.py --compare-initial
```

The root runners execute the self-contained `reproduction/code` pipeline.
It includes 63 numerical tests and eleven experiment drivers. The existing
standalone source directories remain available.

## Run experiments

```sh
# Balanced multidimensional capacity study only
python -I run_numerics.py --prefix run --balanced-only
python -I reproduction/verification/check_balanced_results.py --fresh run_balanced

# All eleven numerical studies
python -I run_numerics.py --prefix all_studies
python -I reproduction/verification/check_reproduction.py --prefix all_studies
python -I reproduction/verification/check_balanced_results.py --fresh all_studies_balanced

# Individual scalar-capacity and curvature studies
python -I reproduction/code/experiments/scalar_capacity.py --output scalar_run
python -I reproduction/code/experiments/smooth_curvature.py --output curvature_run
```

Use a new output prefix. Existing result directories are preserved.

## Included experiments and data

- Scalar coupled capacities: 120 instances, with forward-query budgets of
  9, 33 and 129. At 129 shared queries, the original and adaptive bounds
  detect 5 and 71 of the 113 positive-reference cases.
- Balanced multidimensional capacities: 42 full-rank instances at three
  query budgets with three timing repeats, plus 20 structured conditions
  with three timing repeats, giving 438 timed runs.
- Independent complete-region reference risks for twelve small capacity
  instances, with separate forward-projection checks.
- Rotated folded-ball and affine controls, with explicit structural upper
  risks. These controls test separation values rather than adaptive
  partitioning in general high-dimensional problems.
- Smooth-response curvature, exact parabolic identities, and
  Hamilton-Jacobi disk controls: 37 cases.
- Planar recovery, two-branch geometry, fixed-fiber checks, target
  classification, polygonal path realization, computational separation,
  and anisotropic parameter controls.

`reproduction/code/src/` contains the implementations, `tests/` the analytic
and independent numerical controls, `experiments/` the drivers, and `notes/`
the numerical protocols. Saved synthetic results are under
`reproduction/code/results/`. The two complete balanced executions agree in
scientific values; elapsed times and solver-warning counts can vary.

All query setup, forward solves, anchor updates, upper-bound operations,
conic checks and bookkeeping enter the balanced study's reported totals.
The supplied solver warnings, residuals, duality gaps and numerical backoffs
are retained. These are floating-point diagnostics, not interval validation.
Zero detections are inconclusive. General capacity tests with three, four
or eight latent directions are not presented as successful detections.

The checks use package-local imports and do not require a parent project
directory. Generated figures, numerical summaries and execution logs are
excluded from Git tracking.
