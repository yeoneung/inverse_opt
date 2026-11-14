"""Run the local numerical tests with package-local imports."""
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    from src import optimal_recovery, mixture_recovery, fiber_certificate, polygon_recovery,projection_paths,adaptive_certificate,box_certificate
    from experiments import classification
    for module in [optimal_recovery, mixture_recovery, fiber_certificate, polygon_recovery,projection_paths,adaptive_certificate,box_certificate,classification]:
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise RuntimeError('A scientific module was imported outside this code package.')
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests=result.testsRun, passed=result.wasSuccessful(),
                  seconds=time.perf_counter()-start,
                  scientific_imports_package_local=True,
                  scope='Recovery, fixed and adaptive certificates, global path realization, exact geometry, source normalization and coupled allocation checks.')
    output = ROOT/'results/test_results.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
