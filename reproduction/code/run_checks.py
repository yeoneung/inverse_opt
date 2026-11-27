"""Run the recovery tests included with this paper, without parent imports."""
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    from src import optimal_recovery, mixture_recovery, fiber_certificate, polygon_recovery,projection_paths,adaptive_certificate,box_certificate,consistency_complexity,smooth_curvature,multidimensional_certification
    from experiments import classification_study
    for module in [optimal_recovery, mixture_recovery, fiber_certificate, polygon_recovery,projection_paths,adaptive_certificate,box_certificate,consistency_complexity,smooth_curvature,multidimensional_certification,classification_study]:
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise RuntimeError('A scientific module was imported outside this code package.')
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests=result.testsRun, passed=result.wasSuccessful(),
                  seconds=time.perf_counter()-start,
                  scientific_imports_package_local=True,
                  scope='Recovery, convex certificates, global paths, exact geometry, coupled allocation, computational separation, smooth-curvature/HJB controls, and multidimensional query pipelines with independent global references.')
    output = ROOT.parent/'verification/test_results.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
