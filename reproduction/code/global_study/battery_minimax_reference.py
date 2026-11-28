"""Independent proper-risk intervals on the battery projection graph.

Finite-witness minimax LOWER-bounds the full proper risk. A global worst-risk
UPPER at its feasible candidate upper-bounds full proper risk. Floating-point
global bounds and KKT checks are reported, not interval-certified arithmetic.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import time

ROOT=Path(__file__).resolve().parent
import gurobipy as gp
import numpy as np

from application_battery import physical_schedule,serializable
from global_upper_gurobi import derive_projection_bounds,solve_global_upper
from src.multidimensional_certification import Projector


def status_name(status):
    names=['OPTIMAL','TIME_LIMIT','NODE_LIMIT','SOLUTION_LIMIT','SUBOPTIMAL',
           'NUMERIC','INTERRUPTED','INFEASIBLE','INF_OR_UNBD','UNBOUNDED']
    return {getattr(gp.GRB,name):name.lower() for name in names}.get(status,str(status))


def solve_restricted_witness_minimax(A,b,c,G,witness_parameters,witness_responses,
                                    *,geometry=None,time_limit=60.,log_path=None,
                                    relative_gap=1e-7,absolute_gap=1e-8,
                                    lower_backoff=1e-7,seed=271127):
    """Globally minimize the finite-witness risk on the projection graph.

    A finite-witness incumbent is NOT an upper bound on the full risk.
    Witnesses and candidate policies are independently reprojected. The
    variable bounds come from LPs and a strict Slater certificate.
    """
    started=time.perf_counter()
    A,b,c,G=map(lambda x:np.asarray(x,float),(A,b,c,G))
    witness_parameters=np.asarray(witness_parameters,float)
    supplied=np.asarray(witness_responses,float)
    m,q=A.shape
    p=G.shape[1]
    assert witness_parameters.shape==(len(supplied),p) and supplied.shape[1]==q
    assert np.max(np.abs(witness_parameters))<=1+1e-10
    geometry_fresh=geometry is None
    if geometry_fresh:
        geometry=derive_projection_bounds(A,b,c,G)
    witness_started=time.perf_counter()
    projector=Projector(A,b,q)
    responses=[];multipliers=[]
    for parameter in witness_parameters:
        y,lam=projector(c+G@parameter)
        responses.append(y);multipliers.append(lam)
    points=np.asarray(responses)
    multipliers=np.asarray(multipliers)
    discrepancy=float(np.max(np.abs(points-supplied),initial=0.))
    assert discrepancy<2e-7,discrepancy
    zero_y,zero_lam=projector(c)
    candidates=np.vstack([points,zero_y])
    candidate_parameters=np.vstack([witness_parameters,np.zeros(p)])
    candidate_multipliers=np.vstack([multipliers,zero_lam])
    errors=np.max(np.sum((candidates[:,None,:]-points[None,:,:])**2,axis=2),axis=1)
    index=int(np.argmin(errors))
    warm_v=candidate_parameters[index]
    warm_y=candidates[index]
    warm_lam=candidate_multipliers[index]
    warm_t=float(errors[index])
    witness_seconds=time.perf_counter()-witness_started
    build_started=time.perf_counter()
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0)
        env.start()
        with gp.Model('proper_battery_finite_witness_risk',env=env) as model:
            model.Params.OutputFlag=0
            model.Params.Threads=1
            model.Params.Seed=int(seed)
            model.Params.TimeLimit=float(time_limit)
            model.Params.MIPGap=float(relative_gap)
            model.Params.MIPGapAbs=float(absolute_gap)
            model.Params.FeasibilityTol=1e-8
            model.Params.IntFeasTol=1e-8
            model.Params.OptimalityTol=1e-8
            if log_path:
                model.Params.OutputFlag=1
                model.Params.LogToConsole=0
                model.Params.LogFile=str(log_path)
            v=[model.addVar(lb=-1.,ub=1.,name=f'v{j}') for j in range(p)]
            y=[model.addVar(lb=float(geometry['y_lower'][j]),
                            ub=float(geometry['y_upper'][j]),name=f'y{j}') for j in range(q)]
            lam=[model.addVar(lb=0.,ub=float(geometry['lambda_upper'][i]),
                              name=f'lambda{i}') for i in range(m)]
            active=[model.addVar(vtype=gp.GRB.BINARY,name=f'active{i}') for i in range(m)]
            # A feasible proper candidate proves this finite-objective cap.
            t=model.addVar(lb=0.,ub=warm_t+1e-7*max(1.,warm_t),name='witness_error')
            for i in range(m):
                row=gp.quicksum(float(A[i,j])*y[j] for j in range(q) if A[i,j])
                model.addConstr(row<=float(b[i]),name=f'primal{i}')
                model.addGenConstrIndicator(active[i],False,lam[i]==0.)
                model.addGenConstrIndicator(active[i],True,row==float(b[i]))
            for j in range(q):
                model.addConstr(y[j]-float(c[j])-gp.quicksum(float(G[j,k])*v[k] for k in range(p))
                    +gp.quicksum(float(A[i,j])*lam[i] for i in range(m) if A[i,j])==0,
                    name=f'stationarity{j}')
            for i,point in enumerate(points):
                model.addQConstr(gp.quicksum((y[j]-float(point[j]))*(y[j]-float(point[j])) for j in range(q))<=t,
                                 name=f'witness{i}')
            model.setObjective(t,gp.GRB.MINIMIZE)
            for j in range(p):v[j].Start=float(warm_v[j])
            for j in range(q):y[j].Start=float(warm_y[j])
            for i in range(m):
                lam[i].Start=float(warm_lam[i])
                active[i].Start=int(warm_lam[i]>1e-10)
            t.Start=warm_t
            model.update()
            variable_count=model.NumVars
            linear_count=model.NumConstrs
            quadratic_count=model.NumQConstrs
            indicator_count=model.NumGenConstrs
            build_seconds=time.perf_counter()-build_started
            model.optimize()
            status=int(model.Status)
            solutions=int(model.SolCount)
            solver_seconds=float(model.Runtime)
            nodes=float(model.NodeCount)
            raw_lower=float(model.ObjBound) if status not in [gp.GRB.INFEASIBLE,gp.GRB.INF_OR_UNBD,gp.GRB.UNBOUNDED] else -math.inf
            raw_incumbent=float(model.ObjVal) if solutions else None
            solver_gap=float(model.MIPGap) if solutions else math.inf
            chosen_v=np.asarray([x.X for x in v]) if solutions else warm_v
            chosen_y=np.asarray([x.X for x in y]) if solutions else warm_y
            chosen_lam=np.asarray([x.X for x in lam]) if solutions else warm_lam
    validation_started=time.perf_counter()
    checked_y,checked_lam=projector(c+G@chosen_v)
    checked_finite_risk=float(np.max(np.sum((points-checked_y)**2,axis=1)))
    slack=b-A@chosen_y
    residual=max(float(np.max(A@chosen_y-b,initial=0.)),
        float(np.max(-chosen_lam,initial=0.)),
        float(np.linalg.norm(chosen_y-c-G@chosen_v+A.T@chosen_lam,np.inf)),
        float(np.max(np.abs(chosen_lam*slack),initial=0.)),
        float(np.max(np.abs(chosen_v)-1,initial=0.)),projector.maximum_kkt_residual)
    projection_difference=float(np.linalg.norm(chosen_y-checked_y,np.inf))
    if max(residual,projection_difference)>2e-6:
        raise RuntimeError('Proper-risk incumbent failed independent projection checks.')
    lower=0.
    if math.isfinite(raw_lower) and status!=gp.GRB.NUMERIC:
        lower=max(0.,raw_lower-lower_backoff*max(1.,abs(raw_lower)))
    if lower>checked_finite_risk+2e-6:
        raise RuntimeError('Finite-risk lower bound exceeds its feasible incumbent.')
    warm_data=dict(parameters=witness_parameters,responses=points,multipliers=multipliers,
                   seconds=witness_seconds)
    report=dict(lower=lower,raw_solver_lower=raw_lower,
        finite_witness_incumbent=raw_incumbent,validated_finite_risk=checked_finite_risk,
        status=status_name(status),status_code=status,solutions=solutions,gap=solver_gap,
        nodes=nodes,variables=variable_count,linear_constraints=linear_count,
        quadratic_constraints=quadratic_count,indicator_constraints=indicator_count,
        witness_count=len(points),maximum_witness_projection_difference=discrepancy,
        parameter=chosen_v.tolist(),response=checked_y.tolist(),
        maximum_kkt_residual=residual,projection_difference=projection_difference,
        maximum_checked_multiplier=float(np.max(checked_lam)),
        geometry_seconds=float(geometry['preparation_seconds']) if geometry_fresh else 0.,
        witness_seconds=witness_seconds,build_seconds=build_seconds,solver_seconds=solver_seconds,
        validation_seconds=time.perf_counter()-validation_started,
        total_seconds=time.perf_counter()-started,lower_backoff=lower_backoff,
        slater_margin=geometry['slater_margin'],maximum_dual_bound=float(np.max(geometry['lambda_upper'])),
        solver='Gurobi',solver_version=list(gp.gurobi.version()),threads=1,seed=seed,
        complementarity='indicator',formulation='quadratic_epigraph_on_projection_KKT_graph')
    return report,warm_data


def reference_case(label,history,*,lower_time_limit=60.,upper_time_limit=60.,output=None):
    started=time.perf_counter()
    data=json.loads((ROOT/(label+'.json')).read_text())
    instance=data['instance']
    A,b,c,G=[np.asarray(instance[key],float) for key in ['A','b','c','G']]
    corner=data['corner']
    parameters=np.asarray(corner['parameters'],float)
    responses=np.asarray(corner['responses'],float)
    mask=np.all(np.abs(parameters)>1-1e-10,axis=1)
    parameters=parameters[mask]
    responses=responses[mask]
    for row in history:
        parameters=np.vstack([parameters,np.asarray(row['oracle']['parameter'])])
        responses=np.vstack([responses,np.asarray(row['oracle']['response'])])
    assert len(parameters)==2**G.shape[1]+len(history)
    geometry=derive_projection_bounds(A,b,c,G)
    stem=Path(output).with_suffix('') if output else ROOT/(label+'_proper_reference')
    finite,warm=solve_restricted_witness_minimax(A,b,c,G,parameters,responses,
        geometry=geometry,time_limit=lower_time_limit,log_path=str(stem)+'_minimax.log')
    upper=solve_global_upper(A,b,c,G,np.asarray(finite['response']),
        time_limit=upper_time_limit,geometry=geometry,warm_data=warm,
        complementarity='indicator',objective_form='stationarity',
        relative_gap=1e-7,absolute_gap=1e-8,log_path=str(stem)+'_proper_risk.log')
    last=history[-1]
    R_lower=float(last['variance'])
    R_upper=float(last['oracle']['upper'])
    P_lower=float(finite['lower'])
    P_upper=float(upper['upper'])
    certificate=float(last['certificate'])
    penalty_lower=max(0.,P_lower-R_upper,certificate)
    penalty_upper=max(0.,P_upper-R_lower)
    if penalty_lower>penalty_upper+2e-6:
        raise RuntimeError('Independent risk interval contradicts the certificate.')
    fraction=[certificate/penalty_upper if penalty_upper>0 else None,
              min(1.,certificate/penalty_lower) if penalty_lower>0 else None]
    physical=physical_schedule(instance,np.asarray(finite['response']))
    physical_A=np.asarray(instance['physical_A'])
    physical_b=np.asarray(instance['physical_b'])
    violation=float(np.max(physical_A@physical['power_kw']-physical_b,initial=0.))
    assert violation<2e-6
    return dict(label=label,p=G.shape[1],q=G.shape[0],witnesses=len(parameters),
        proper_risk_lower=P_lower,proper_risk_upper=P_upper,
        mixed_risk_lower=R_lower,mixed_risk_upper=R_upper,
        actual_penalty_lower=penalty_lower,actual_penalty_upper=penalty_upper,
        algorithm_certificate=certificate,captured_fraction_interval=fraction,
        restricted_witness_minimax=finite,proper_action_worst_risk=upper,
        proper_physical_schedule=serializable(physical),
        mixed_physical_schedule=serializable(physical_schedule(instance,np.asarray(last['anchor']))),
        maximum_physical_feasibility_residual=violation,
        geometry_preparation_seconds=geometry['preparation_seconds'],
        total_seconds=time.perf_counter()-started,
        scope='Finite-witness global minimization lower-bounds P; a separate global worst-response solve upper-bounds P at a feasible proper candidate. Floating-point bounds with numerical backoffs, not interval validation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lower-time-limit',type=float,default=60.)
    parser.add_argument('--upper-time-limit',type=float,default=60.)
    args=parser.parse_args()
    assert not args.output.exists(),'Preserve existing reference output.'
    histories=json.loads((ROOT/'battery_exchange_pilot.json').read_text())
    rows=[]
    for entry in histories:
        if entry['label'] not in ['battery_corner_q8_p4_cosine','battery_corner_q8_p4_cosine_mixed']:
            continue
        print('Reference '+entry['label'],flush=True)
        rows.append(reference_case(entry['label'],entry['history'],lower_time_limit=args.lower_time_limit,
            upper_time_limit=args.upper_time_limit,output=args.output.with_name(args.output.stem+'_'+entry['label'])))
        print(json.dumps({k:rows[-1][k] for k in ['label','proper_risk_lower','proper_risk_upper',
            'actual_penalty_lower','actual_penalty_upper','captured_fraction_interval','total_seconds']},indent=2),flush=True)
    args.output.write_text(json.dumps(dict(rows=rows,passed=True),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
