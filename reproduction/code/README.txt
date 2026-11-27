Quadratic projection and minimax recovery experiments

Use Python 3.12 and install code/requirements.txt. From the package root:
  python -I code/run_checks.py
  python -I code/run_numerics.py --prefix NEW_NAME --balanced-only
  python -I verification/check_balanced_results.py --fresh NEW_NAME_balanced
This runs the new multidimensional study. All eleven studies can be run by
omitting --balanced-only. Then run verification/check_reproduction.py --prefix
NEW_NAME as well. Fresh simple result names are required.

NEW STUDY
src/multidimensional_certification.py: reusable forward QPs with KKT
polishing, finite sampled enclosing balls, original/adaptive SOCP checks,
adaptive query boxes and local affine upper-risk bounds, exhaustive
small-dimensional global references, and rotated folded-ball controls.
experiments/balanced_capacity.py: full protocol and all timing stages.
notes/balanced_capacity_protocol.json: dimensions, seeds, scales,
query budgets, repetitions and numerical acceptance thresholds.
results/balanced_reproduction: primary raw runs, instance arrays,
warnings, environment and summary. balanced_initial retains the
first independent execution. Scientific values agree; elapsed times vary.
experiments/report_balanced_base.py retains the complete numerical report and Fig3.
experiments/report_scalar_studies.py generates the scalar calibration in Fig2.
experiments/report_jogo.py renders the JGO presentation with every condition retained.
experiments/report_jogo.py renders the journal-facing exposition with
exact-risk controls first. It uses the same complete raw runs, retains
all experimental conditions, and changes no table values.
verification/balanced_table_values.json records the numerical table values.

GLOBAL REFERENCES
The small reference enumerates every independent target active subset,
solves its critical-region inequalities, enumerates vertices and solves
all region minimax problems. It does not reuse the query method's samples
or anchor. It is intended for small cases and is not an optimized global
solver benchmark. Independent forward probes and analytic fold controls
check the reference. Exact-arithmetic statements and floating-point
acceptance tests are distinguished throughout.

NUMERICAL ACCEPTANCE AND COSTS
All query setup, forward solves, sampled-anchor updates, upper-risk work,
SOCP checks and bookkeeping enter the reported totals. Python imports and
instance generation are excluded. Each method receives the instance as
input; no ground-truth center or region partition is passed to it.
The structured controls explicitly use their special upper-risk formula.
Some conic solves warn about inaccurate solutions. Those warnings are
retained; dual feasibility, primal/dual gaps and numerical backoffs are
recorded separately. This is not interval-certified arithmetic.

PRECEDING STUDIES
The ten preceding studies and their supplied data are retained: planar
recovery, mixtures, fixed-fiber checks, structural classification, polygons,
projection paths, coupled capacity, anisotropic multidimensional boxes,
computational separation, and smooth curvature. The anisotropic study is
an exploratory retained control and is not evidence for the new balanced
latent-dimension experiment. Code/run_numerics.py lists all entry points.
Renderers for preceding studies write verification/rendered_outputs;
report_balanced_base.py updates the retained numerical section and Fig3;
report_jogo.py generates the active JGO section from the same complete data.
The reproduction checker permits numerical tolerance and excludes timings.

HEADLINE SCALAR STUDY
experiments/scalar_capacity.py reproduces the 120 scalar capacity cases.
At 129 shared queries the original and adaptive bounds detect 5 and 71
positive cases, respectively. The folded-ball study is an implementation
control with an analytical upper bound; it does not test adaptive partitioning.
