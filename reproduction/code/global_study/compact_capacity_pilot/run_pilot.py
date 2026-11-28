"""Bounded exploratory global-solver pilot with eight latent directions."""
from pathlib import Path
import hashlib
import json
import os
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
    result_path=OUTPUT/'results.json'
    if result_path.exists():
        raise FileExistsError('Preserve pilot results: '+str(result_path))
    protocol={'q':12,'p':8,'singular_value':0.75,
        'seeds':[846100,846101,846102],'capacity_factors':[0.7,1.4],
        'global_seconds_per_call':15.0,'maximum_exchanges':6,
        'capacity_multiplier_bounds':True,
        'first_case':[846100,0.7],
        'continuation_rule':'Run remaining five cases only if first case detects a positive certificate; retain all outcomes.',
        'scope':'Exploratory tractability pilot; a new smaller-action family, separate from original q32 tests.'}
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    report={'complete':False,'protocol':protocol,'cases':[],'failures':[]}
    started=time.perf_counter()
    for seed in protocol['seeds']:
        for factor in protocol['capacity_factors']:
            label=f'q12_p8_s{seed}_c{factor}'
            A,b,c,G,design=instance(12,8,seed,factor,.75)
            arrays=OUTPUT/(label+'_instance.npz')
            np.savez_compressed(arrays,A=A,b=b,c=c,G=G)
            try:
                begin=time.perf_counter()
                result=run_exchange(A,b,c,G,label='compact_capacity_pilot/'+label,
                    time_limit=15.,iterations=6,capacity_bounds=True)
                result.update(instance=label,seed=seed,capacity_factor=factor,
                    design=design,instance_file=arrays.name,
                    instance_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),
                    elapsed_seconds=time.perf_counter()-begin)
                report['cases'].append(result)
                print('COMPLETED',label,'detected',result['detected'],'certificate',result['certificate'],
                    'iterations',len(result['history']),'total_seconds',result['total_seconds'],flush=True)
            except Exception as error:
                report['failures'].append({'instance':label,'error':type(error).__name__+': '+str(error)})
                (OUTPUT/(label+'_failure.log')).write_text(traceback.format_exc(),encoding='utf-8')
                print('FAILED',label,type(error).__name__,str(error),flush=True)
            report['wall_seconds']=time.perf_counter()-started
            result_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
            if seed==846100 and factor==0.7 and (report['failures'] or not report['cases'][0]['detected']):
                report['continuation_stopped']='First compact pilot did not detect a positive certificate within the fixed budget.'
                report['complete']=True
                result_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
                return
    report['complete']=True
    report['detections']=sum(row['detected'] for row in report['cases'])
    report['wall_seconds']=time.perf_counter()-started
    result_path.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
