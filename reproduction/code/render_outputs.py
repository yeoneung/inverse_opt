"""Recreate the JGO data section and figure from the complete retained study."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment['PYTHONUTF8'] = '1'
    for name in ['report_scalar_studies.py','report_consistency_separation.py','report_smooth_curvature.py','report_balanced_base.py','report_jogo.py']:
        subprocess.run([sys.executable,'-I',str(ROOT/'experiments'/name)],
                       cwd=ROOT,env=environment,check=True)
    print('Rendered the JGO result section and Figures 2 and 3. Historical cross-checks are in verification/rendered_outputs/.')


if __name__ == '__main__':
    main()
