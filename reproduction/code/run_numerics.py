"""Recompute every final experiment into new directories; saved data are preserved."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='reproduction', help='New simple directory prefix under code/results.')
    parser.add_argument('--balanced-only', action='store_true', help='Run only the new multidimensional balanced-capacity study.')
    args = parser.parse_args()
    if not args.prefix or Path(args.prefix).name != args.prefix or any(c in args.prefix for c in '/\\:'):
        raise ValueError('Use a simple output prefix')
    if args.balanced_only:
        subprocess.run([sys.executable,'-I',str(ROOT/'experiments/balanced_capacity.py'),
                        '--output',args.prefix+'_balanced'],cwd=ROOT,check=True)
        return
    outputs = {name:args.prefix+'_'+name for name in ['recovery','mixture','fiber','classification','polygon','paths','capacity','multidimensional','complexity','curvature']}
    for name in outputs.values():
        if (ROOT/'results'/name).exists():
            raise FileExistsError('Choose a fresh prefix; preserving '+name)
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment['PYTHONUTF8'] = '1'
    environment['OPENBLAS_NUM_THREADS'] = '1'
    environment['OMP_NUM_THREADS'] = '1'
    log_directory = ROOT/'results'/(args.prefix+'_logs')
    log_directory.mkdir()
    timings = {}
    for name, script in [('recovery','planar_recovery.py'),
                         ('mixture','mixture_study.py'),('fiber','fiber_study.py'),('classification','classification_study.py'),('polygon','polygon_study.py'),
                         ('paths','projection_path_study.py'),('capacity','scalar_capacity.py'),
                         ('multidimensional','anisotropic_controls.py'),
                         ('complexity','consistency_separation.py'),
                         ('curvature','smooth_curvature.py')]:
        command = [sys.executable, '-I', str(ROOT/'experiments'/script), '--output', outputs[name]]
        if name == 'mixture':
            command += ['--coupled-source',outputs['recovery']]
        started = time.perf_counter()
        with (log_directory/(name+'.log')).open('w', encoding='utf-8') as log:
            completed = subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        timings[name] = time.perf_counter()-started
        if completed.returncode:
            raise RuntimeError('Experiment failed; see '+str(log_directory/(name+'.log')))
        if name=='complexity':
            with (log_directory/'complexity_exact.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-I',str(ROOT/'experiments/exact_separation_controls.py'),
                    '--output',str(ROOT/'results'/outputs[name]/'exact_controls.json')],
                    cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(name+' complete: '+outputs[name],flush=True)
    begin=time.perf_counter()
    outputs['balanced']=args.prefix+'_balanced'
    subprocess.run([sys.executable,'-I',str(ROOT/'experiments/balanced_capacity.py'),
                    '--output',outputs['balanced']],cwd=ROOT,env=environment,check=True)
    timings['balanced']=time.perf_counter()-begin
    report = dict(prefix=args.prefix,outputs=outputs,seconds=timings,
                  note='Fresh computations. Runtime differences are expected; copied historical solver stages are not rerun.')
    (log_directory/'reproduction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
