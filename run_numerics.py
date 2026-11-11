"""Run the seven numerical experiments in new output directories."""
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
    parser.add_argument('--prefix', default='run', help='New simple directory prefix under results/.')
    args = parser.parse_args()
    if not args.prefix or Path(args.prefix).name != args.prefix or any(c in args.prefix for c in '/\\:'):
        raise ValueError('Use a simple output prefix')
    outputs = {name:args.prefix+'_'+name for name in ['recovery','mixture','fiber','classification','polygon','paths','capacity']}
    for name in outputs.values():
        if (ROOT/'results'/name).exists():
            raise FileExistsError('Choose a fresh prefix; preserving '+name)
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment['PYTHONUTF8'] = '1'
    environment['OPENBLAS_NUM_THREADS'] = '1'
    environment['OMP_NUM_THREADS'] = '1'
    log_directory = ROOT/'results'/(args.prefix+'_logs')
    log_directory.mkdir(parents=True)
    timings = {}
    for name, script in [('recovery','recovery.py'),
                         ('mixture','mixture.py'),('fiber','fiber.py'),('classification','classification.py'),('polygon','polygon.py'),
                         ('paths','global_paths.py'),('capacity','capacity.py')]:
        command = [sys.executable, '-I', str(ROOT/'experiments'/script), '--output', outputs[name]]
        if name == 'mixture':
            command += ['--coupled-source',outputs['recovery']]
        started = time.perf_counter()
        with (log_directory/(name+'.log')).open('w', encoding='utf-8') as log:
            completed = subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        timings[name] = time.perf_counter()-started
        if completed.returncode:
            raise RuntimeError('Experiment failed; see '+str(log_directory/(name+'.log')))
        print(name+' complete: '+outputs[name],flush=True)
    report = dict(prefix=args.prefix,outputs=outputs,seconds=timings,
                  note='Fresh computations. Runtime differences across machines are expected.')
    (log_directory/'reproduction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
