"""Tight numerical mixed/proper risk intervals for the battery application.

The mixed reference iterates finite enclosing balls and globally bounded
worst-response problems without a certificate-detection stopping rule.
Existing input and output artifacts are read-only. A final independent
proper minimax solve uses all discovered witness responses.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np

from application_battery import physical_schedule,serializable
from battery_minimax_reference import solve_restricted_witness_minimax
from global_upper_gurobi import derive_projection_bounds,solve_global_upper
from src.multidimensional_certification import Projector,sampled_ball,distance_checks

ROOT=Path(__file__).resolve().parent


def tight_case(entry,*,time_limit=15.,iterations=15,tolerance=1e-6,stem):
    started=time.perf_counter()
    source=json.loads((ROOT/(entry['label']+'.json')).read_text())
    instance=source['instance']
    A,b,c,G=[np.asarray(instance[key],float) for key in ['A','b','c','G']]
    Q=np.asarray(instance['Q'])
    corner=source['corner']
    parameters=np.asarray(corner['parameters'],float)
    responses=np.asarray(corner['responses'],float)
    mask=np.all(np.abs(parameters)>1-1e-10,axis=1)
    parameters=parameters[mask]
    supplied=responses[mask]
    for row in entry['history']:
        parameters=np.vstack([parameters,np.asarray(row['oracle']['parameter'])])
        supplied=np.vstack([supplied,np.asarray(row['oracle']['response'])])
    geometry=derive_projection_bounds(A,b,c,G)
    projector=Projector(A,b,len(c))
    witness_begin=time.perf_counter()
    responses=[];multipliers=[]
    for parameter in parameters:
        y,lam=projector(c+G@parameter)
        responses.append(y);multipliers.append(lam)
    responses=np.asarray(responses)
    multipliers=np.asarray(multipliers)
    witness_disagreement=float(np.max(np.abs(responses-supplied)))
    assert witness_disagreement<2e-7
    witness_seconds=time.perf_counter()-witness_begin
    warm=dict(parameters=parameters,responses=responses,multipliers=multipliers,seconds=witness_seconds)
    history=[]
    mixed_lower=0.
    mixed_upper=float('inf')
    best_anchor=None
    best_oracle=None
    anchor_seconds=oracle_seconds=separation_seconds=0.
    stopping='iteration_limit'
    for iteration in range(iterations):
        begin=time.perf_counter()
        ball=sampled_ball(warm['responses'])
        anchor_seconds+=time.perf_counter()-begin
        begin=time.perf_counter()
        oracle=solve_global_upper(A,b,c,G,ball['anchor'],time_limit=time_limit,
            geometry=geometry,warm_data=warm,complementarity='indicator',objective_form='stationarity',
            relative_gap=1e-7,absolute_gap=1e-8,target_upper=None,
            log_path=str(stem)+f'_mixed_iteration{iteration}.log')
        oracle_seconds+=time.perf_counter()-begin
        mixed_lower=max(mixed_lower,float(ball['variance']))
        if oracle['upper']<mixed_upper:
            mixed_upper=float(oracle['upper'])
            best_anchor=np.asarray(ball['anchor']).copy()
            best_oracle=oracle
        history.append(dict(iteration=iteration,witnesses=len(warm['responses']),
            anchor=ball['anchor'].tolist(),variance=ball['variance'],sample_upper=ball['sample_upper'],
            mixed_lower=mixed_lower,mixed_upper=mixed_upper,gap=mixed_upper-mixed_lower,oracle=oracle))
        print(entry['label'],'mixed iteration',iteration,'gap',mixed_upper-mixed_lower,
              'status',oracle['status'],flush=True)
        parameter=np.asarray(oracle['parameter'])
        y,lam=projector(c+G@parameter)
        duplicate=float(np.min(np.sum((warm['responses']-y)**2,axis=1)))<1e-18
        if not duplicate:
            warm['parameters']=np.vstack([warm['parameters'],parameter])
            warm['responses']=np.vstack([warm['responses'],y])
            warm['multipliers']=np.vstack([warm['multipliers'],lam])
        if mixed_upper-mixed_lower<=tolerance:
            stopping='global_risk_interval_tolerance'
            break
        if duplicate:
            stopping='no_new_witness_within_oracle_budget'
            break
    assert mixed_lower<=mixed_upper+2e-6
    begin=time.perf_counter()
    final_separation=distance_checks(G,c,A,b,best_anchor)
    separation_seconds+=time.perf_counter()-begin
    proper,warm_reference=solve_restricted_witness_minimax(A,b,c,G,
        warm['parameters'],warm['responses'],geometry=geometry,time_limit=time_limit,
        log_path=str(stem)+'_proper_minimax.log')
    proper_upper=solve_global_upper(A,b,c,G,np.asarray(proper['response']),
        geometry=geometry,warm_data=warm_reference,time_limit=time_limit,
        complementarity='indicator',objective_form='stationarity',
        relative_gap=1e-7,absolute_gap=1e-8,
        log_path=str(stem)+'_proper_worst_risk.log')
    P_lower=float(proper['lower']);P_upper=float(proper_upper['upper'])
    certificate=float(entry['certificate'])
    penalty_lower=max(0.,P_lower-mixed_upper)
    penalty_upper=max(0.,P_upper-mixed_lower)
    assert certificate<=penalty_upper+2e-6
    assert penalty_lower<=penalty_upper+2e-6
    captured=[certificate/penalty_upper if penalty_upper>0 else None,
              min(1.,certificate/penalty_lower) if penalty_lower>0 else None]
    mixed_physical=physical_schedule(instance,best_anchor)
    proper_physical=physical_schedule(instance,np.asarray(proper['response']))
    schedule_difference=np.asarray(proper_physical['power_kw'])-np.asarray(mixed_physical['power_kw'])
    whitened_difference=np.asarray(proper['response'])-best_anchor
    action_difference_squared=float(schedule_difference@Q@schedule_difference)
    assert abs(action_difference_squared-whitened_difference@whitened_difference)<1e-10
    physical_A=np.asarray(instance['physical_A']);physical_b=np.asarray(instance['physical_b'])
    max_violation=max(float(np.max(physical_A@np.asarray(s['power_kw'])-physical_b,initial=0.))
                      for s in [mixed_physical,proper_physical])
    assert max_violation<2e-6
    return dict(label=entry['label'],p=G.shape[1],q=G.shape[0],
        mixed_risk_lower=mixed_lower,mixed_risk_upper=mixed_upper,
        mixed_reference_gap=mixed_upper-mixed_lower,mixed_reference_converged=mixed_upper-mixed_lower<=tolerance,
        proper_risk_lower=P_lower,proper_risk_upper=P_upper,
        actual_penalty_lower=penalty_lower,actual_penalty_upper=penalty_upper,
        squared_risk_ratio_interval=[P_lower/mixed_upper,P_upper/mixed_lower],
        algorithm_certificate=certificate,captured_fraction_interval=captured,
        mixed_history=history,mixed_stopping_reason=stopping,
        mixed_anchor=best_anchor.tolist(),proper_parameter=proper['parameter'],proper_action=proper['response'],
        mixed_physical_schedule=serializable(mixed_physical),proper_physical_schedule=serializable(proper_physical),
        mixed_worst_response_schedule=serializable(physical_schedule(instance,np.asarray(best_oracle['response']))),
        proper_worst_response_schedule=serializable(physical_schedule(instance,np.asarray(proper_upper['response']))),
        Q_weighted_center_difference=action_difference_squared,
        maximum_physical_constraint_violation=max_violation,
        refined_anchor_separation=final_separation,
        proper_finite_minimax=proper,proper_worst_risk=proper_upper,
        witness_count_for_proper_reference=len(warm['responses']),
        initial_witness_projection_disagreement=witness_disagreement,
        geometry_seconds=geometry['preparation_seconds'],witness_seconds=witness_seconds,
        mixed_anchor_seconds=anchor_seconds,mixed_oracle_seconds=oracle_seconds,
        refined_separation_seconds=separation_seconds,total_seconds=time.perf_counter()-started,
        scope='Independent risk intervals use global dual bounds and genuine finite witnesses. Numerical backoffs are recorded; no interval-arithmetic certification is claimed. No certificate-based early stopping is used in the mixed reference.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'battery_tight_reference.json')
    parser.add_argument('--time-limit',type=float,default=15.)
    parser.add_argument('--iterations',type=int,default=15)
    parser.add_argument('--tolerance',type=float,default=1e-6)
    args=parser.parse_args()
    assert not args.output.exists(),'Preserve previous outputs.'
    entries=json.loads((ROOT/'battery_exchange_pilot.json').read_text())
    rows=[]
    for entry in entries:
        if entry['label'] not in ['battery_corner_q8_p4_cosine','battery_corner_q8_p4_cosine_mixed']:
            continue
        stem=args.output.with_name(args.output.stem+'_'+entry['label']).with_suffix('')
        row=tight_case(entry,time_limit=args.time_limit,iterations=args.iterations,
                       tolerance=args.tolerance,stem=stem)
        rows.append(row)
        print(json.dumps({k:row[k] for k in ['label','mixed_risk_lower','mixed_risk_upper',
            'proper_risk_lower','proper_risk_upper','actual_penalty_lower','actual_penalty_upper',
            'captured_fraction_interval','total_seconds']},indent=2),flush=True)
    args.output.write_text(json.dumps(dict(passed=True,rows=rows),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
