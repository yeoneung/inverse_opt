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
Zero detections are inconclusive. The original 513-query capacity benchmark
misses the tested higher-dimensional cases; the global-solver extension below
uses additional computation and witness updates.

The checks use package-local imports and do not require a parent project
directory. Generated figures, numerical summaries and execution logs are
excluded from Git tracking.


## Global worst-response exchange and battery scheduling

The portable `reproduction/code/global_study/` directory adds KKT-based global
upper-risk optimization, witness/center exchange and complete synthetic records:

- The twelve existing four-parameter capacity instances all yield positive
  lower bounds, compared with none for the original 513-query benchmark.
  Eight are detected after corner initialization and one global call;
  only four need additional responses.
- A separate exploratory eight-parameter, twelve-action cohort detects five of
  six conditions. Its initial failed pilot, extended budget, final failure and
  the unresolved larger eight-parameter pilot are retained.
- An idealized 24-hour battery model detects seven of nine conditions.
  Independent global reference risks quantify the captured prediction penalty.
- An independent forward-QP/SOCP audit reconstructs witnesses and anchors in
  all 27 new cases and checks 92 sampled anchors. Conservative upper-bound
  transfer accounts for numerical anchor repairs.

Check the recorded data without optional global solvers:

```sh
python -I reproduction/verification/check_global_results.py
```

Optional global runs and the independent QP/SOCP audit require the documented
Gurobi/PySCIPOpt dependencies. See `reproduction/code/global_study/README.txt`
for installation, license scope and commands with fresh output paths. Original
records are preserved; only private machine-path headers in three logs were
redacted. No solver binaries are bundled. The new detection timings include
independent validation and exclude separate tight-reference computations.
The experiments do not match the query benchmark's runtime budget or claim a
speed advantage. Corner initialization costs `2**p` forward solves. The battery
uses synthetic prices/loads and a lossless model; its risk is a quadratic
schedule-prediction error, not a monetary loss. All bounds use floating-point
solvers and numerical checks, not interval validation.


The saved-run stage report reproduces first-call counts of 2/6, 6/6 and 0/6
for the two four-parameter cohorts and the eight-parameter cohort. Median
global-call counts are 2, 1 and 8.5, respectively. It analyzes preserved
histories and independent anchor checks, without adding a timed ablation:

```sh
python -I reproduction/code/global_study/report_exchange_stages.py
```


## Render numerical reports

```sh
python -I reproduction/code/experiments/report_jogo.py
```

The renderer separates compact numerical results from detailed computational
settings and checks, and retains the folded-ball control report. All three
outputs use the same complete raw records. Numerical calculations and table
values are unchanged. Generated report files remain outside Git tracking.
