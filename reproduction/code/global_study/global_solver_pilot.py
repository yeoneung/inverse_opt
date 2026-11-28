"""Run bounded-budget global upper-risk pilots on three fixed capacity cases."""
from pathlib import Path
import json
import os
import sys
import time

for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
from global_upper_gurobi import derive_projection_bounds,corner_witnesses,solve_global_upper


def main():
    output=ROOT/'global_solver_pilot.json'
    if output.exists():
        raise FileExistsError(output)
    corners=json.loads((ROOT/'corner_pilot_results.json').read_text())['cases']
    chosen=[corners[0],corners[6],corners[12]]
    arrays=np.load(ROOT.parent/'results/balanced_reproduction/instances.npz')
    results=[]
    for row in chosen:
        name=row['instance']
        A,b,c,G=[arrays[name+'_'+key] for key in ['A','b','c','G']]
        geometry=derive_projection_bounds(A,b,c,G)
        warm=corner_witnesses(A,b,c,G)
        target=row['variance']+row['adaptive_distance']**2-1e-6
        for kind,objective in [('indicator','squared'),('bigm','stationarity')]:
            begin=time.perf_counter()
            solved=solve_global_upper(A,b,c,G,np.array(row['anchor']),time_limit=20,
                geometry=geometry,warm_data=warm,complementarity=kind,
                objective_form=objective,target_upper=target,
                log_path=ROOT/(name+'_'+kind+'_'+objective+'.log'))
            solved.update(instance=name,p=row['p'],q=row['q'],variance=row['variance'],
                adaptive_distance=row['adaptive_distance'],target_upper=target,
                certificate=max(0.,row['variance']+row['adaptive_distance']**2-solved['upper']))
            results.append(solved)
            output.write_text(json.dumps({'complete':len(results)==6,'cases':results},indent=2),encoding='utf-8')
            print(name,kind,objective,'UB',solved['upper'],'LB',solved['lower'],
                  'threshold',target,'certificate',solved['certificate'],'seconds',time.perf_counter()-begin,flush=True)


if __name__=='__main__':
    main()
