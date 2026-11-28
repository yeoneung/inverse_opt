"""Pilot witness refresh before global-risk optimization; retain every case."""
from itertools import product
from pathlib import Path
import json
import os
import sys
import time

for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'

ROOT=Path(__file__).resolve().parent
PACKAGE=ROOT.parent
sys.path.insert(0,str(PACKAGE))
import numpy as np
from src.multidimensional_certification import Projector,sampled_ball,distance_checks


def main():
    output=ROOT/'corner_pilot_results.json'
    if output.exists():
        raise FileExistsError(output)
    report=json.loads((PACKAGE/'results/balanced_reproduction/runs.json').read_text())
    arrays=np.load(PACKAGE/'results/balanced_reproduction/instances.npz')
    rows=[r for r in report['rows'] if r['kind']=='capacity' and r['budget']==513 and r['p'] in [4,8] and r['repetition']==0]
    results=[]
    for row in rows:
        name=row['instance']
        A,b,c,G=[arrays[name+'_'+key] for key in ['A','b','c','G']]
        begin=time.perf_counter()
        projector=Projector(A,b,len(c))
        latent=np.asarray(list(product([-1.,1.],repeat=G.shape[1])))
        points=np.array([projector(c+G @ v)[0] for v in latent])
        forward=time.perf_counter()-begin
        old_anchor=np.array(row['anchor'])
        old_corner_lower=float(np.max(np.sum((points-old_anchor)**2,axis=1)))
        begin=time.perf_counter()
        ball=sampled_ball(points)
        anchor_seconds=time.perf_counter()-begin
        begin=time.perf_counter()
        checks=distance_checks(G,c,A,b,ball['anchor'])
        separation_seconds=time.perf_counter()-begin
        entry=dict(instance=name,p=row['p'],q=row['q'],seed=row['seed'],capacity_factor=row['capacity_factor'],
            previous_variance=row['variance'],previous_distance=row['adaptive_distance'],
            previous_corner_risk_lower=old_corner_lower,
            previous_anchor_unrescuable_by_U_only=old_corner_lower>=row['variance']+row['adaptive_distance']**2,
            corner_queries=len(points),anchor=ball['anchor'].tolist(),variance=ball['variance'],sample_upper=ball['sample_upper'],
            adaptive_distance=checks['adaptive_distance'],original_distance=checks['original_distance'],
            corner_optimistic_gap=ball['variance']+checks['adaptive_distance']**2-ball['sample_upper'],
            forward_seconds=forward,anchor_seconds=anchor_seconds,separation_seconds=separation_seconds,
            max_kkt_residual=projector.maximum_kkt_residual)
        results.append(entry)
        output.write_text(json.dumps({'cases':results,'complete':len(results)==len(rows)},indent=2),encoding='utf-8')
        print(name,'corner optimistic gap',entry['corner_optimistic_gap'],'distance2',checks['adaptive_distance']**2,flush=True)


if __name__=='__main__':
    main()
