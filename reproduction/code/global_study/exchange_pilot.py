"""Global worst-response exchange with complete stage accounting."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parent))
import numpy as np
from global_upper_gurobi import derive_projection_bounds,corner_witnesses,solve_global_upper
from src.multidimensional_certification import Projector,sampled_ball,distance_checks


def run_exchange(A,b,c,G,*,label,time_limit=15.,iterations=6,geometry=None,capacity_bounds=False,log_directory=None):
    start=time.perf_counter()
    geometry_start=time.perf_counter()
    geometry=derive_projection_bounds(A,b,c,G) if geometry is None else geometry
    if capacity_bounds:
        from capacity_dual_bounds import capacity_multiplier_bounds
        strengthened=capacity_multiplier_bounds(A,b,c,G)
        geometry['lambda_upper']=np.minimum(geometry['lambda_upper'],strengthened['lambda_upper'])
    geometry_seconds=time.perf_counter()-geometry_start
    warm=corner_witnesses(A,b,c,G)
    samples=warm['responses'].copy()
    projector=Projector(A,b,len(c))
    history=[];anchor_seconds=distance_seconds=oracle_seconds=validation_seconds=0.
    stopping_reason='iteration_budget'
    log_root=ROOT if log_directory is None else Path(log_directory)
    log_stem=label if log_directory is None else Path(label).name
    (log_root/Path(log_stem).parent).mkdir(parents=True,exist_ok=True)
    for iteration in range(iterations):
        begin=time.perf_counter();ball=sampled_ball(samples);anchor_seconds+=time.perf_counter()-begin
        begin=time.perf_counter();checks=distance_checks(G,c,A,b,ball['anchor']);distance_seconds+=time.perf_counter()-begin
        threshold=ball['variance']+checks['adaptive_distance']**2-1e-6
        begin=time.perf_counter()
        solved=solve_global_upper(A,b,c,G,ball['anchor'],time_limit=time_limit,
            geometry=geometry,warm_data=warm,complementarity='bigm',objective_form='stationarity',
            target_upper=threshold,log_path=log_root/(log_stem+f'_iteration{iteration}.log'))
        oracle_seconds+=time.perf_counter()-begin
        certificate=max(0.,ball['variance']+checks['adaptive_distance']**2-solved['upper'])
        entry=dict(iteration=iteration,anchor=ball['anchor'].tolist(),variance=ball['variance'],
            sample_upper=ball['sample_upper'],adaptive_distance=checks['adaptive_distance'],
            original_distance=checks['original_distance'],certificate=certificate,oracle=solved)
        history.append(entry)
        print(label,'iteration',iteration,'nu',ball['variance'],'d2',checks['adaptive_distance']**2,
              'UB',solved['upper'],'LB',solved['lower'],'certificate',certificate,flush=True)
        if certificate>1e-6:
            stopping_reason='positive_certificate'
            break
        begin=time.perf_counter();y,lam=projector(c+G @ np.array(solved['parameter']));validation_seconds+=time.perf_counter()-begin
        if float(np.min(np.sum((samples-y)**2,axis=1)))<1e-18:
            stopping_reason='no_new_witness_within_oracle_budget'
            break
        improvement=float(np.sum((y-ball['anchor'])**2)-ball['sample_upper'])
        if improvement<1e-7:
            stopping_reason='no_improving_witness_within_oracle_budget'
            break
        samples=np.vstack([samples,y])
        warm['parameters']=np.vstack([warm['parameters'],solved['parameter']])
        warm['responses']=samples
        warm['multipliers']=np.vstack([warm['multipliers'],lam])
    return dict(label=label,p=G.shape[1],q=G.shape[0],history=history,detected=history[-1]['certificate']>1e-6,
        certificate=history[-1]['certificate'],geometry_seconds=geometry_seconds,
        initial_query_seconds=warm['seconds'],initial_queries=2**G.shape[1],
        additional_forward_queries=projector.calls+sum(h['oracle']['solutions']>0 for h in history),
        stopping_reason=stopping_reason,anchor_seconds=anchor_seconds,distance_seconds=distance_seconds,
        oracle_seconds=oracle_seconds,validation_seconds=validation_seconds,total_seconds=time.perf_counter()-start)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['capacity','battery'],required=True)
    args=parser.parse_args()
    output=ROOT/(args.kind+'_exchange_pilot.json')
    if output.exists():raise FileExistsError(output)
    results=[]
    if args.kind=='capacity':
        arrays=np.load(ROOT.parent/'results/balanced_reproduction/instances.npz')
        for label in ['q16_p4_s846100_c0.7','q32_p4_s846100_c0.7','q32_p8_s846100_c0.7']:
            A,b,c,G=[arrays[label+'_'+key] for key in ['A','b','c','G']]
            result=run_exchange(A,b,c,G,label='exchange_'+label,capacity_bounds=True)
            results.append(result);output.write_text(json.dumps(results,indent=2),encoding='utf-8')
    else:
        for name in ['battery_corner_q8_p4_cosine.json','battery_corner_q8_p4_cosine_mixed.json']:
            source=json.loads((ROOT/name).read_text());data=source['instance']
            A,b,c,G=[np.array(data[key]) for key in ['A','b','c','G']]
            result=run_exchange(A,b,c,G,label=Path(name).stem,time_limit=15)
            result['physical_checks']=source['checks']
            results.append(result);output.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
