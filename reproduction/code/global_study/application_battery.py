"""Synthetic battery scheduling application for a quadratic inverse model.

This is a development prototype, not a calibrated market-data study. Positive
power charges a lossless battery. The model deliberately separates known
tariffs and device limits from unknown desired reserve-energy trajectories.
It supplies a mathematically verified box of source-compatible objectives.

The reusable forward/certificate implementation is imported from the existing
local reproducibility package. No existing package file is modified.
"""
from pathlib import Path
from itertools import product
import argparse
import json
import sys
import time

import cvxpy as cp
import numpy as np

PACKAGE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PACKAGE))
from src.multidimensional_certification import (Projector, query_certificate,
                                               sampled_ball,distance_checks)


def battery_instance(p=4, *, horizon=8, reserve_basis='blocks',
                     target_power_kw=4., terminal_min_kwh=4.,
                     reserve_amplitude_kwh=4., cycling_weight=.04,
                     tracking_weight=.005):
    """Return physical and whitened instances with a source-fit certificate.

    The default eight 3-hour slots represent a synthetic day. Under the block
    basis, four latent coordinates control two reserve targets each; eight
    control one each. Nonconstant cosine modes alternatively model uncertainty
    in reserve timing while holding average desired energy fixed at 16 kWh.
    Their coefficients are scaled so every target stays within 16 +/- amplitude.
    Reserve targets stay in [12,20] kWh at the default amplitude. The known
    source tariff is a synthetic expensive event. The target tariff is a
    synthetic time-of-use day; neither tariff is represented as measured data.
    """
    T=int(horizon)
    if not 1<=p<=T or T<4:
        raise ValueError('Use at most one reserve parameter per time slot.')
    if not 0<=terminal_min_kwh<=32:
        raise ValueError('Final reserve must lie within the energy capacity.')
    step=24./T
    initial=16.
    capacity=32.
    minimum=4.
    source_power=.5
    grid_cap=5.
    midpoints=(np.arange(T)+.5)*step
    original_midpoints=(np.arange(8)+.5)*3
    load=np.interp(midpoints,original_midpoints,[1.8,1.6,1.8,2.5,3.6,4.3,3.2,2.4])
    target_tariff=np.interp(midpoints,original_midpoints,[.09,.07,.06,.10,.20,.35,.24,.14])
    source_tariff=np.full(T,1.2)
    cumulative=np.tril(np.ones((T,T)))
    if reserve_basis=='blocks':
        basis=np.zeros((T,p))
        basis[np.arange(T),np.minimum(p-1,np.arange(T)*p//T)]=1.
    elif reserve_basis in ['cosine','cosine_mixed']:
        modes=np.arange(1,p+1) if reserve_basis=='cosine' else 2*np.arange(p)+1
        if np.max(modes)>=T:
            raise ValueError('Nonconstant cosine modes must be below the horizon.')
        basis=np.sqrt(2/T)*np.cos(np.pi*(np.arange(T)[:,None]+.5)*modes[None,:]/T)
        basis/=np.max(np.sum(np.abs(basis),axis=1))
        assert np.max(np.abs(np.sum(basis,axis=0)))<1e-12
    else:
        raise ValueError('Unknown reserve basis.')
    reserve_base=np.full(T,16.)
    reserve_map=reserve_amplitude_kwh*basis
    Q=cycling_weight*np.eye(T)+tracking_weight*step**2*cumulative.T@cumulative
    W=np.linalg.cholesky(Q).T
    Winv=np.linalg.solve(W,np.eye(T))
    preferred_linear=tracking_weight*step*cumulative.T@reserve_map
    target_b0=tracking_weight*step*cumulative.T@(reserve_base-initial)-step*target_tariff
    source_b0=tracking_weight*step*cumulative.T@(reserve_base-initial)-step*source_tariff

    def constraints(power, final_reserve):
        # No grid export and a site import cap are merged into the power box.
        lower=-np.minimum(power,load)
        upper=np.minimum(power,grid_cap-load)
        energy_lower=np.full(T,minimum)
        energy_lower[-1]=max(minimum,final_reserve)
        A=np.vstack([-np.eye(T),np.eye(T),-step*cumulative,step*cumulative])
        b=np.r_[-lower,upper,initial-energy_lower,np.full(T,capacity-initial)]
        return A,b,lower,upper

    physical_A,physical_b,power_lower,power_upper=constraints(target_power_kw,terminal_min_kwh)
    source_A,source_b,_,_=constraints(source_power,minimum)
    observed_power=np.full(T,-source_power)
    # At this observed lower-power corner, nonnegative gradient components
    # are multipliers on -x <= source_power. Other multipliers can be zero.
    source_gradient_center=Q@observed_power-source_b0
    source_gradient_radius=np.sum(np.abs(preferred_linear),axis=1)
    source_margin=float(np.min(source_gradient_center-source_gradient_radius))
    source_feasibility=float(np.max(source_A@observed_power-source_b))
    if source_margin<1e-10 or source_feasibility>1e-10:
        raise ValueError('This configuration does not certify the full prior box.')
    target_A=physical_A@Winv
    c=np.linalg.solve(W.T,target_b0)
    G=np.linalg.solve(W.T,preferred_linear)
    source_c=np.linalg.solve(W.T,source_b0)
    source_A_white=source_A@Winv
    y_observed=W@observed_power
    # A finite box for whitened actions is derived from the physical power box.
    positive=np.maximum(W,0.)
    negative=np.minimum(W,0.)
    white_lower=positive@power_lower+negative@power_upper
    white_upper=positive@power_upper+negative@power_lower
    return dict(A=target_A,b=physical_b,c=c,G=G,Q=Q,W=W,
                physical_A=physical_A,physical_b=physical_b,
                power_lower=power_lower,power_upper=power_upper,
                white_action_lower=white_lower,white_action_upper=white_upper,
                source_A=source_A_white,source_b=source_b,source_c=source_c,
                source_observation=y_observed,source_power_observation=observed_power,
                source_gradient_center=source_gradient_center,
                source_gradient_radius=source_gradient_radius,
                source_fit_margin=source_margin,
                source_feasibility_residual=source_feasibility,
                target_b0=target_b0,preferred_linear=preferred_linear,
                reserve_base=reserve_base,reserve_map=reserve_map,
                step_hours=step,initial_energy_kwh=initial,capacity_kwh=capacity,
                minimum_energy_kwh=minimum,terminal_minimum_kwh=terminal_min_kwh,
                source_converter_kw=source_power,target_converter_kw=target_power_kw,
                grid_import_cap_kw=grid_cap,load_kw=load,
                target_tariff_usd_per_kwh=target_tariff,
                source_tariff_usd_per_kwh=source_tariff,
                cycling_weight=cycling_weight,tracking_weight=tracking_weight,
                dimension=T,latent_dimension=p,reserve_basis=reserve_basis,
                slot_midpoints_hours=midpoints,reserve_parameter_basis=basis,
                assumptions=['Synthetic tariffs, load and reserve target ranges.',
                    'Ideal lossless battery; no simultaneous charge/discharge variables.',
                    'Positive signed power charges; grid import equals load plus power.',
                    'Known strictly convex cycling and state-tracking quadratic metric.',
                    'Unknown reserve targets, constant across known source/target contexts.',
                    'No source noise: the entire declared prior box fits the source observation.',
                    'Whitened squared-action risk is a Q-weighted schedule error, not electricity bill regret.'])


def physical_schedule(instance,white_action):
    power=np.linalg.solve(instance['W'],np.asarray(white_action))
    soc=instance['initial_energy_kwh']+instance['step_hours']*np.cumsum(power)
    return dict(power_kw=power,soc_kwh=soc,grid_import_kw=instance['load_kw']+power)


def checks(instance,seed=87011):
    """Check source compatibility and whitening against an independent physical QP."""
    T,p=instance['G'].shape
    rng=np.random.default_rng(seed)
    source=Projector(instance['source_A'],instance['source_b'],T)
    target=Projector(instance['A'],instance['b'],T)
    source_error=whitening_error=feasibility=0.
    probes=[np.zeros(p),np.ones(p),-np.ones(p),*rng.uniform(-1.,1.,(8,p))]
    for v in probes:
        ys,_=source(instance['source_c']+instance['G']@v)
        source_error=max(source_error,float(np.linalg.norm(ys-instance['source_observation'],np.inf)))
        y,_=target(instance['c']+instance['G']@v)
        x=cp.Variable(T)
        problem=cp.Problem(cp.Minimize(.5*cp.quad_form(x,instance['Q'])
            -(instance['target_b0']+instance['preferred_linear']@v)@x),
            [instance['physical_A']@x<=instance['physical_b']])
        problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
        if problem.status not in ['optimal','optimal_inaccurate']:
            raise RuntimeError('Independent physical QP failed.')
        whitening_error=max(whitening_error,float(np.linalg.norm(y-instance['W']@x.value,np.inf)))
        physical=physical_schedule(instance,y)
        feasibility=max(feasibility,float(np.max(instance['physical_A']@physical['power_kw']-instance['physical_b'])))
    assert source_error<2e-6
    assert whitening_error<2e-5
    assert feasibility<2e-7
    return dict(passed=True,probes=len(probes),source_fit_margin=instance['source_fit_margin'],
                maximum_source_response_error=source_error,
                maximum_independent_whitening_error=whitening_error,
                maximum_physical_feasibility_residual=feasibility,
                maximum_forward_kkt_residual=max(source.maximum_kkt_residual,target.maximum_kkt_residual),
                latent_rank=int(np.linalg.matrix_rank(instance['G'])),
                latent_singular_values=np.linalg.svd(instance['G'],compute_uv=False).tolist())


def serializable(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,dict):
        return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,list):
        return [serializable(v) for v in value]
    return value


def corner_witness_certificate(instance):
    """A finite witness/anchor calculation; no global upper-risk oracle.

    All 2**p box corners are genuine compatible parameters. Their sampled
    enclosing-ball anchor and conic separation may be passed to an independent
    global upper-risk solver. A positive separation alone is NOT a positive
    risk-penalty certificate away from the true mixed center.
    """
    started=time.perf_counter()
    p=instance['latent_dimension']
    parameters=np.vstack([np.zeros(p),np.asarray(list(product([-1.,1.],repeat=p)))])
    projector=Projector(instance['A'],instance['b'],instance['dimension'])
    responses=np.asarray([projector(instance['c']+instance['G']@v)[0] for v in parameters])
    forward_finished=time.perf_counter()
    ball=sampled_ball(responses)
    anchor_finished=time.perf_counter()
    distances=distance_checks(instance['G'],instance['c'],instance['A'],instance['b'],ball['anchor'])
    separation_finished=time.perf_counter()
    return dict(parameters=parameters,responses=responses,anchor=ball['anchor'],
                variance=ball['variance'],sample_upper=ball['sample_upper'],**distances,
                queries=len(parameters),forward_seconds=forward_finished-started,
                anchor_seconds=anchor_finished-forward_finished,
                distance_seconds=separation_finished-anchor_finished,
                total_seconds=separation_finished-started,
                maximum_forward_kkt_residual=projector.maximum_kkt_residual)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--budgets',type=int,nargs='*',default=[33,129])
    parser.add_argument('--dimensions',type=int,nargs='*',default=[4,8])
    parser.add_argument('--horizon',type=int,default=8)
    parser.add_argument('--reserve-basis',choices=['blocks','cosine','cosine_mixed'],default='blocks')
    parser.add_argument('--reserve-amplitude',type=float,default=4.)
    parser.add_argument('--corner-witnesses',action='store_true',
                        help='Also save sampled anchors from zero and every prior-box corner.')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    corners=[]
    started=time.perf_counter()
    for p in args.dimensions:
        instance=battery_instance(p,horizon=args.horizon,reserve_basis=args.reserve_basis,
                                  reserve_amplitude_kwh=args.reserve_amplitude)
        verified=checks(instance)
        (args.output/f'instance_p{p}.json').write_text(json.dumps(serializable(instance),indent=2),encoding='utf-8')
        if args.corner_witnesses:
            corner=corner_witness_certificate(instance)
            corners.append(dict(p=p,q=instance['dimension'],instance_checks=verified,**corner))
            (args.output/f'corner_witness_p{p}.json').write_text(json.dumps(serializable(corner),indent=2),encoding='utf-8')
        corner=corner_witness_certificate(instance)
        (args.output/f'corner_p{p}.json').write_text(json.dumps(serializable(corner),indent=2),encoding='utf-8')
        for budget in args.budgets:
            print(f'Battery p={p}, budget={budget}',flush=True)
            row=query_certificate(instance['A'],instance['b'],instance['c'],instance['G'],budget)
            row.update(application='synthetic_battery',p=p,q=instance['dimension'],budget=budget,instance_checks=verified)
            row['anchor_schedule']=serializable(physical_schedule(instance,np.asarray(row['anchor'])))
            rows.append(row)
    report=dict(passed=True,rows=rows,corners=corners,seconds=time.perf_counter()-started,
                scope='Prototype only. Query lower bounds may be zero. No global-solver upper bound is computed here.')
    (args.output/'runs.json').write_text(json.dumps(serializable(report),indent=2),encoding='utf-8')
    print(json.dumps([dict(p=r['p'],budget=r['budget'],variance=r['variance'],
        upper=r['upper'],adaptive_distance=r['adaptive_distance'],adaptive_gap=r['adaptive_gap']) for r in rows],indent=2))


if __name__=='__main__':
    main()
