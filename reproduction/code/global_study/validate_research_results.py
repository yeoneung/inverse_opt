"""Independent forward, contact-distribution, and SOCP checks of saved runs."""
from itertools import product
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
from scipy.optimize import linprog
from src.multidimensional_certification import distance_checks,sampled_ball
from projection_audit import GurobiAuditProjector


def validate_case(run,A,b,c,G):
    start=time.perf_counter()
    projector=GurobiAuditProjector(A,b)
    points=np.asarray([projector(c+G @ np.asarray(v))[0] for v in product([-1.,1.],repeat=G.shape[1])])
    preparation_seconds=time.perf_counter()-start
    checks=[]
    for entry in run['history']:
        x=np.asarray(entry['anchor'])
        objective=np.sum((points-x)**2,axis=1)
        contact=linprog(-objective,A_eq=np.vstack([np.ones(len(points)),points.T]),
                        b_eq=np.r_[1.,x],bounds=(0.,None),method='highs')
        contact_mean=contact.x @ points if contact.success else None
        actual_variance=float(contact.x @ objective) if contact.success else None
        conic=distance_checks(G,c,A,b,x)
        assert abs(conic['adaptive_distance']-entry['adaptive_distance'])<3e-6
        assert conic['maximum_dual_residual']<2e-7
        assert conic['maximum_distance_gap']<2e-6
        oracle=entry['oracle']
        v=np.asarray(oracle['parameter'])
        prior_violation=max(float(np.max(v-1.)),float(np.max(-1.-v)),0.)
        assert prior_violation<2e-7
        v=np.clip(v,-1.,1.)
        forward_start=time.perf_counter()
        response,_=projector(c+G @ v)
        preparation_seconds+=time.perf_counter()-forward_start
        response_difference=float(np.linalg.norm(response-np.asarray(oracle['response']),np.inf))
        lower=float(np.sum((response-x)**2))
        assert oracle['upper']>=lower-2e-6
        recomputed=max(0.,entry['variance']+conic['adaptive_distance']**2-oracle['upper'])
        assert abs(recomputed-entry['certificate'])<3e-6
        assert oracle['maximum_kkt_residual']<2e-6
        checks.append(dict(iteration=entry['iteration'],
            contact_mean_residual=float(np.max(np.abs(contact_mean-x))) if contact.success else None,
            contact_variance=actual_variance,reported_variance=entry['variance'],
            original_anchor_has_exact_contact_representation=contact.success,
            stored_response_vs_independent_QP=response_difference,
            maximum_dual_residual=conic['maximum_dual_residual'],
            maximum_distance_gap=conic['maximum_distance_gap'],
            saved_vs_recomputed_distance=abs(conic['adaptive_distance']-entry['adaptive_distance']),
            prior_violation=prior_violation,independent_feasible_risk=lower,
            upper_risk=oracle['upper'],recomputed_certificate=recomputed))
        if entry['iteration']<len(run['history'])-1:
            points=np.vstack([points,response])
    # Use only independently projected compatible points for the final certificate.
    # This certifies the result even when a historical interior-point witness
    # had a small weak-face error. The old global bound transfers by the triangle
    # inequality; it is never replaced by a feasible incumbent.
    final_start=time.perf_counter()
    fresh=sampled_ball(points)
    old=run['history'][-1]
    shift=float(np.linalg.norm(fresh['anchor']-np.asarray(old['anchor'])))
    transferred_upper=float((np.sqrt(old['oracle']['upper'])+shift)**2)
    conic=distance_checks(G,c,A,b,fresh['anchor'])
    assert conic['maximum_dual_residual']<2e-7
    assert conic['maximum_distance_gap']<2e-6
    certificate=float(max(0.,fresh['variance']+conic['adaptive_distance']**2-transferred_upper))
    final_seconds=time.perf_counter()-final_start
    residual=projector.maximum_kkt_residual
    projector.close()
    return dict(label=run['label'],passed=True,detected=certificate>1e-6,
                original_detected=run['detected'],validated_certificate=certificate,
                validated_anchor=fresh['anchor'].tolist(),validated_variance=fresh['variance'],
                validated_distance=conic['adaptive_distance'],anchor_shift=shift,
                transferred_global_upper=transferred_upper,
                independent_repair_seconds=preparation_seconds+final_seconds,
                original_pipeline_seconds=run['total_seconds'],
                pipeline_and_repair_seconds=run['total_seconds']+preparation_seconds+final_seconds,
                maximum_independent_projection_kkt_residual=residual,anchors=checks)


def validate_source(data):
    A,b,c,G,Q,W=[np.asarray(data[key]) for key in ['source_A','source_b','source_c','G','Q','W']]
    power=np.asarray(data['source_power_observation'])
    source=np.asarray(data['source_observation'])
    assert np.max(np.abs(W @ power-source))<1e-9
    assert np.max(np.abs(Q-W.T @ W))<1e-9
    gradient=Q @ power-W.T @ c
    radius=np.abs(W.T @ G).sum(axis=1)
    margin=float(np.min(gradient-radius))
    assert margin>=-1e-9
    assert np.max(A @ source-b)<1e-8
    # Physical lower-power rows establish x >= source_power throughout X_source.
    q=len(power)
    assert np.max(np.abs(A[:q] @ W+np.eye(q)))<1e-8
    assert np.max(np.abs(b[:q]+power))<1e-8
    return margin


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reruns'/'independent_result_checks.json')
    options=parser.parse_args()
    if options.output.exists():raise FileExistsError(options.output)
    options.output.parent.mkdir(parents=True,exist_ok=True)
    with np.load(ROOT.parent/'results/balanced_reproduction/instances.npz') as archive:
        arrays={name:archive[name] for name in archive.files}
    p4=json.loads((ROOT/'capacity_p4_validation.json').read_text())
    output=[]
    for row in p4['cases']:
        name=row['label'].removeprefix('p4validation_')
        args=[arrays[name+'_'+key] for key in ['A','b','c','G']]
        output.append(validate_case(row,*args))
    battery=json.loads((ROOT/'battery_grid_validation/runs.json').read_text())
    for row in battery['rows']:
        data=row['instance']
        source_margin=validate_source(data)
        check=validate_case(row['exchange'],*[np.asarray(data[key]) for key in ['A','b','c','G']])
        check['universal_source_gradient_margin']=source_margin
        output.append(check)
    compact=json.loads((ROOT/'compact_capacity_extended/results.json').read_text())
    for row in compact['cases']:
        with np.load(ROOT/'compact_capacity_extended'/row['instance_file']) as data:
            output.append(validate_case(row,*[data[key] for key in ['A','b','c','G']]))
    summary={'passed':True,'cases':len(output),'anchors':sum(len(row['anchors']) for row in output),
        'detected':sum(row['detected'] for row in output),'capacity_p4_complete':p4['complete'],
        'compact_p8_complete':compact['complete'],'compact_p8_cases':len(compact['cases']),
        'battery_cases':len(battery['rows']),'method':'Independent Gurobi primal-simplex forward QPs, simplex contact LP, repeated dual SOCP checks, and analytical whole-prior source gradients.',
        'final_certificate_method':'Rebuild the feasible witness cloud and its anchor; transfer the global upper by (sqrt(U_old)+anchor_shift)^2 and repeat conic separation.',
        'rows':output}
    options.output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
