# Inverse optimization numerics

Numerical routines for polyhedral projections, image-restricted and unrestricted
recovery, exact polygonal examples, and convex normal-cone distance certificates.

Use Python 3.12:

```sh
python -m pip install -r requirements.txt
python -I run_checks.py
python -I run_numerics.py --prefix run
```

The test runner executes 43 local checks. The numerical runner executes eight
experiments and creates CSV/JSON outputs under `results/`. Existing output
directories are preserved; choose another prefix to run a new batch.

## Layout

- `src/`: recovery routines, normal-cone certificates, projection paths, and exact geometry.
- `experiments/`: deterministic experiment drivers.
- `configs/`: numerical parameters, random seeds, tolerances, and query budgets.
- `tests/`: analytic controls and independent numerical checks.

## Individual experiments

```sh
python -I experiments/recovery.py --output example_recovery
python -I experiments/mixture.py --output example_mixture --coupled-source example_recovery
python -I experiments/fiber.py --output example_fiber
python -I experiments/classification.py --output example_classification
python -I experiments/polygon.py --output example_polygon
python -I experiments/global_paths.py --output example_paths
python -I experiments/capacity.py --output example_capacity
python -I experiments/multidimensional.py --output example_multidimensional
```

The capacity experiment uses 120 synthetic cases in action dimensions 4, 8, 16,
and 32, with forward-query budgets of 9, 33, and 129. Numerical workers use one
BLAS/OMP thread. Timings and solver-warning counts can vary across machines.

Finite witness points must be model responses. Anchors use their contact means,
and finite-query upper values include a Lipschitz covering term. Each adaptive
radius evaluation solves primal and dual conic problems. Floating-point solver
diagnostics and conservative numerical backoffs do not constitute interval
validation. Generated outputs and local environments are excluded from version
control.

The multidimensional experiment uses 135 paired conditions with two, four, or
eight parameters and five nuisance radii. It reports forward-query, contact-anchor,
upper-bound and convex-checking times separately. Its validation uses scalar
central paths and Hausdorff envelopes, without computing a full multidimensional
response partition. Large nuisance radii and loose covering bounds can leave the
certificates uninformative. All specified conditions and solver warnings are retained.
