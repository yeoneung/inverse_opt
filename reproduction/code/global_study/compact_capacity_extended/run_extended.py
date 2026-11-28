"""Larger exchange budget for the preserved eight-parameter compact pilot."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parent))
import numpy as np
from exchange_pilot import run_exchange
from experiments.balanced_capacity import instance


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reruns'/'compact_capacity_extended')
    args=parser.parse_args()
    OUTPUT=args.output.resolve()
    OUTPUT.mkdir(parents=True,exist_ok=False)
    result_path=OUTPUT/'results.json'
    if result_path.exists():
        raise FileExistsError(result_path)
    old=json.loads((ROOT/'compact_capacity_pilot/results.json').read_text())
    assert len(old['cases'])==1 and not old['cases'][0]['detected']
    protocol={'q':12,'p':8,'singular_value':0.75,'seeds':[846100,846101,846102],
        'capacity_factors':[0.7,1.4],'global_seconds_per_call':15.,'maximum_exchanges':20,
        'capacity_multiplier_bounds':True,'preserved_initial_run':'compact_capacity_pilot/results.json',
        'prior_six_exchange_result':'No detection; complete previous history and logs retained.',
        'continuation_rule':'If first case detects, continue remaining five; retain all outcomes and stop only at completed-case boundaries if total wall time exceeds 600 seconds.',
        'total_wall_guard_seconds':600.,
        'scope':'Exploratory compact-action q12,p8 family; separate from original q32 capacity family.'}
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    report={'complete':False,'protocol':protocol,'cases':[],'failures':[]}
    start=time.perf_counter()
    for seed in protocol['seeds']:
        for factor in protocol['capacity_factors']:
            label=f'q12_p8_s{seed}_c{factor}'
            A,b,c,G,design=instance(12,8,seed,factor,0.75)
            arrays=OUTPUT/(label+'_instance.npz')
            np.savez_compressed(arrays,A=A,b=b,c=c,G=G)
            try:
                result=run_exchange(A,b,c,G,label='compact_capacity_extended/'+label,
                    time_limit=15.,iterations=20,capacity_bounds=True,log_directory=OUTPUT)
                result.update(instance=label,seed=seed,capacity_factor=factor,
                    design=design,instance_file=arrays.name,
                    instance_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest())
                report['cases'].append(result)
                print('COMPLETED',label,'detected',result['detected'],'certificate',result['certificate'],
                    'calls',len(result['history']),'seconds',result['total_seconds'],flush=True)
            except Exception as error:
                report['failures'].append({'instance':label,'error':type(error).__name__+': '+str(error)})
                (OUTPUT/(label+'_failure.log')).write_text(traceback.format_exc(),encoding='utf-8')
                print('FAILED',label,type(error).__name__,str(error),flush=True)
            report['wall_seconds']=time.perf_counter()-start
            report['detections']=sum(row['detected'] for row in report['cases'])
            result_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
            if seed==846100 and factor==0.7 and (report['failures'] or not report['cases'][0]['detected']):
                report['stopping_reason']='First extended case did not detect within20 exchanges.'
                report['complete']=True
                result_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
                return
            if report['wall_seconds']>protocol['total_wall_guard_seconds']:
                report['stopping_reason']='Exploratory completed-case wall-time guard.'
                report['complete']=True
                result_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
                return
    report['complete']=True
    report['stopping_reason']='All six planned conditions completed.'
    report['wall_seconds']=time.perf_counter()-start
    result_path.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
