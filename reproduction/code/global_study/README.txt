Global worst-response bounds, witness exchange and battery scheduling

This directory contains the code and complete numerical records of the global
solver study. Its imports use the enclosing numerical code directory, which
supplies src/, experiments/ and results/balanced_reproduction/. Keep this
directory inside that code package. No private project paths or bundled solver
binaries are needed.

From the parent code directory, install the numerical and optional solver
requirements with:
  python -m pip install -r global_study/requirements-global.txt

The recorded calculations used Gurobi 14.0.0 and PySCIPOpt 6.3.0 / SCIP 10.0.2.
PySCIPOpt's binary distribution provides SCIP. Gurobi is a separate optional
dependency: a suitable license is required. The included size-limited license
handled the recorded instances. Larger formulations may need another license.
The two solver backends are independently implemented; only Gurobi is needed
for the final independent forward-QP/SOCP revalidation. An optional historical
projection diagnostic also uses OSQP, but it is not required by that validator.

From the submission/reproduction root:
  python verification/check_global_results.py
This reads the recorded results, checks hashes, full latent rank, bound
arithmetic, counts and timings. It does not solve optimization problems.

From this directory:
  python render_research_summary.py --output rendered
  python validate_research_results.py --output reruns/independent_checks.json
  python capacity_p4_validation.py --output reruns/capacity_p4
  python compact_capacity_extended/run_extended.py --output reruns/capacity_p8
  python battery_grid_validation.py --output reruns/battery_grid
  python battery_tight_reference.py --output reruns/battery_reference.json

Choose a fresh output path. Drivers refuse to overwrite preserved results.
Some exploratory pilot commands use fixed output names and intentionally
refuse a rerun in place. Their functions remain importable for new pilots.
The two reference drivers read the preserved battery_exchange_pilot.json
histories and battery_corner_* input files. The validator reads the recorded
cohorts and independently reconstructs all final witnesses and anchors.

The original numerical JSON/JSONL/NPZ records are unchanged. Three log files
have local machine path prefixes redacted; their numerical lines are retained.
Python files have import-path and output-path portability edits.
artifact_manifest.json records original and portable file hashes. A fresh
solver run can differ in elapsed time, search path
and early-stopped certificates. A revalidation uses newly measured validation
time; it must not replace the paper's recorded runtime measurements. The saved
summary's manuscript_modified field describes the historical exploratory run,
not the status of a subsequently edited manuscript.

Study scope and retained negative cases:
- Twelve existing four-parameter capacity cases: all twelve detected.
- A separate q=12,p=8 family: five of six detected after an exploratory budget
  extension. The initial six-exchange failure and the continuation rule are
  preserved in compact_capacity_pilot/ and compact_capacity_extended/protocol.json.
  These were not preregistered experiments. All six extended conditions remain.
- The original q=32,p=8 representative remains inconclusive after a separate
  60-second solve. Its data and solver log are preserved.
- Nine synthetic ideal-battery scenarios: seven detected. These are not field
  data. The reported risk is squared action error in the quadratic metric, not
  a monetary operating loss.

The global maximization solver's dual upper bound is the risk upper bound.
A feasible incumbent supplies only a lower risk. Initializing from all 2**p
corners has exponential dimension dependence. No equal-runtime speedup is
claimed. The independent validator repairs weak-face projection errors and
transfers the old global bound by the triangle inequality before recomputing
the lower certificate. All values use floating-point optimization and numerical
backoffs, not interval arithmetic or machine-checked proof.
