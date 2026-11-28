"""Global worst-response risk by projection KKT and nonconvex mixed-integer QP.

The finite multiplier bounds follow from an actual strict Slater point;
they are never guessed. The maximization solver's ObjBound is the upper
risk. A feasible incumbent is only a lower risk. Numerical diagnostics and
backoffs are recorded; this is floating-point global optimization, not
interval-certified arithmetic.
"""
from itertools import product
from pathlib import Path
import hashlib
import json
import math
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))

import gurobipy as gp
import numpy as np
from scipy.optimize import linprog

from src.multidimensional_certification import Projector


def _fingerprint(A,b,c,G,lo,hi):
    digest=hashlib.sha256()
    for value in [A,b,c,G,lo,hi]:
        array=np.asarray(value,dtype='<f8')
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def derive_projection_bounds(A,b,c,G,v_lower=None,v_upper=None):
    """Bound y and all KKT multipliers using LPs and strict feasibility.

    If ybar is strictly feasible and s_i=b_i-A_i ybar>0, strong convexity
    and KKT give lambda_i*s_i <= .5||ybar-c-Gv||^2. The maximum of this
    convex quadratic over a parameter box occurs at one of its vertices.
    Thus lambda_i <= C/s_i. All dimensions here are small enough to
    enumerate those vertices (the main experiments use p<=8).
    """
    begin=time.perf_counter()
    A,b,c,G=map(lambda x:np.asarray(x,dtype=float),(A,b,c,G))
    m,q=A.shape
    p=G.shape[1]
    lo=np.full(p,-1.) if v_lower is None else np.asarray(v_lower,dtype=float)
    hi=np.full(p,1.) if v_upper is None else np.asarray(v_upper,dtype=float)
    if p>12:
        raise ValueError('This exact corner-based bound preparation is limited to p<=12.')
    if not (b.shape==(m,) and c.shape==(q,) and G.shape==(q,p) and lo.shape==(p,) and hi.shape==(p,)):
        raise ValueError('Inconsistent projection-model dimensions.')
    if np.any(lo>hi):
        raise ValueError('Empty parameter box.')
    free=[(None,None)]*q
    lower=[];upper=[]
    for j in range(q):
        objective=np.eye(q)[j]
        minimum=linprog(objective,A_ub=A,b_ub=b,bounds=free,method='highs')
        maximum=linprog(-objective,A_ub=A,b_ub=b,bounds=free,method='highs')
        if not minimum.success or not maximum.success:
            raise ValueError('A nonempty bounded target is required for the global model.')
        pad=1e-8*max(1.,abs(minimum.fun),abs(maximum.fun))
        lower.append(float(minimum.fun-pad))
        upper.append(float(-maximum.fun+pad))
    row_norm=np.linalg.norm(A,axis=1)
    if np.any(row_norm==0):
        raise ValueError('Remove zero target rows before constructing the KKT model.')
    cheb=linprog(np.r_[np.zeros(q),-1.],A_ub=np.c_[A,row_norm],b_ub=b,
                bounds=free+[(0.,None)],method='highs')
    if not cheb.success:
        raise ValueError('Strict-feasibility LP failed.')
    ybar=cheb.x[:q]
    slack=b-A@ybar
    if np.min(slack/row_norm)<=1e-8:
        raise ValueError('A verified strict Slater point is required; eliminate affine equalities first.')
    corners=np.asarray(list(product(*zip(lo,hi))))
    differences=ybar-c-corners@G.T
    C=float(.5*np.max(np.sum(differences*differences,axis=1)))
    C+=1e-8*max(1.,C)
    dual_upper=C/slack+1e-8
    yl,yu=np.asarray(lower),np.asarray(upper)
    structural_tightening=None
    exact_zero_multipliers=[]
    if (len(b)>=2*q and np.array_equal(A[:q],-np.eye(q))
            and np.array_equal(A[q:2*q],np.eye(q)) and np.array_equal(b[:q],np.zeros(q))
            and np.all(A[2*q:]>=0) and np.all(b[2*q:]>0)
            and np.array_equal(lo,-np.ones(p)) and np.array_equal(hi,np.ones(p))):
        from capacity_dual_bounds import capacity_multiplier_bounds
        structural=capacity_multiplier_bounds(A,b,c,G)
        dual_upper=np.minimum(dual_upper,structural['lambda_upper'])
        preferred_upper=c+np.sum(np.abs(G),axis=1)
        # If the preferred point is strictly below an individual upper
        # bound, nonnegative shared multipliers cannot make that row bind.
        upper_zero=np.flatnonzero(preferred_upper<b[q:2*q]-1e-9)
        dual_upper[q+upper_zero]=0.
        exact_zero_multipliers.extend((q+upper_zero).tolist())
        # D*relu(z) is convex in v for D>=0. Its maximum is attained at
        # parameter-box corners and bounds D*clip(z-D.T eta,0,u).
        preferred=c+corners@G.T
        shared_upper=np.max(np.maximum(preferred,0.)@A[2*q:].T,axis=0) if len(b)>2*q else np.empty(0)
        shared_zero=np.flatnonzero(shared_upper<b[2*q:]-1e-9)
        dual_upper[2*q+shared_zero]=0.
        exact_zero_multipliers.extend((2*q+shared_zero).tolist())
        # For nonnegative shared capacities, y=clip(z-D.T eta,0,u)
        # implies y_j<=max(z_j,0). This bound uses parameter ranges too.
        yu=np.minimum(yu,np.maximum(c+np.sum(np.abs(G),axis=1),0.)+1e-8)
        structural_tightening='nonnegative_capacity_stationarity'
    slack_upper=b-np.maximum(A,0)@yl-np.minimum(A,0)@yu
    slack_upper=np.maximum(slack_upper,0)+1e-8*np.maximum(1.,np.abs(slack_upper))
    return dict(y_lower=yl,y_upper=yu,lambda_upper=dual_upper,slack_upper=slack_upper,
                slater_point=ybar,slater_slack=slack,slater_margin=float(np.min(slack/row_norm)),
                slater_objective_upper=C,parameter_lower=lo,parameter_upper=hi,
                structural_tightening=structural_tightening,exact_zero_multipliers=exact_zero_multipliers,
                fingerprint=_fingerprint(A,b,c,G,lo,hi),preparation_seconds=time.perf_counter()-begin)


def corner_witnesses(A,b,c,G,v_lower=None,v_upper=None):
    """Feasible forward responses at all box corners; no global claim is made."""
    begin=time.perf_counter()
    A,b,c,G=map(lambda x:np.asarray(x,dtype=float),(A,b,c,G))
    p=G.shape[1]
    if p>12:
        raise ValueError('Corner warm starts are limited to p<=12.')
    lo=np.full(p,-1.) if v_lower is None else np.asarray(v_lower,dtype=float)
    hi=np.full(p,1.) if v_upper is None else np.asarray(v_upper,dtype=float)
    parameters=np.asarray(list(product(*zip(lo,hi))))
    projector=Projector(A,b,G.shape[0])
    responses=[];duals=[]
    for v in parameters:
        y,lam=projector(c+G@v)
        responses.append(y);duals.append(lam)
    return dict(parameters=parameters,responses=np.asarray(responses),multipliers=np.asarray(duals),
                forward_residual=projector.maximum_kkt_residual,queries=projector.calls,
                seconds=time.perf_counter()-begin)


def _validate_witnesses(data,A,b,c,G,lo,hi):
    parameters=np.asarray(data['parameters'],float)
    responses=np.asarray(data['responses'],float)
    multipliers=np.asarray(data['multipliers'],float)
    if not len(parameters) or parameters.shape[1:]!=(len(lo),):
        raise ValueError('Warm parameters have the wrong shape or are empty.')
    if np.any(parameters<lo) or np.any(parameters>hi):
        raise ValueError('Warm parameters must belong to the prior exactly.')
    if responses.shape!=(len(parameters),len(c)) or multipliers.shape!=(len(parameters),len(b)):
        raise ValueError('Warm responses/multipliers have the wrong shape.')
    slack=b-responses@A.T
    stationarity=responses-c-parameters@G.T+multipliers@A
    residual=max(float(np.max(-slack,initial=0.)),float(np.max(-multipliers,initial=0.)),
                 float(np.max(np.abs(stationarity),initial=0.)),
                 float(np.max(np.abs(multipliers*slack),initial=0.)))
    if not np.isfinite(residual) or residual>2e-7:
        raise RuntimeError('Warm witnesses fail independent prior/KKT validation.')
    return residual


def solve_global_upper(A,b,c,G,anchor,time_limit=60.,*,geometry=None,warm_data=None,
                       v_lower=None,v_upper=None,complementarity='bigm',
                       objective_form='squared',seed=271127,log_path=None,
                       relative_gap=1e-6,absolute_gap=1e-7,upper_backoff=1e-7,target_upper=None,
                       pwl_segments=32,capacity_structure_cut=False):
    """Return a global upper risk, feasible lower risk, and full solver diagnostics.

    The variable count is p+q+2m. In particular q=32,p=8,m=80 has exactly
    200 variables, without objective/slack auxiliary variables.
    ``geometry`` and ``warm_data`` can be reused across anchor exchange
    iterations. Their preparation costs remain separately visible.
    """
    started=time.perf_counter()
    A,b,c,G,anchor=map(lambda x:np.asarray(x,dtype=float),(A,b,c,G,anchor))
    m,q=A.shape;p=G.shape[1]
    lo=np.full(p,-1.) if v_lower is None else np.asarray(v_lower,dtype=float)
    hi=np.full(p,1.) if v_upper is None else np.asarray(v_upper,dtype=float)
    fresh_geometry=geometry is None
    if fresh_geometry:
        geometry=derive_projection_bounds(A,b,c,G,lo,hi)
    if geometry['fingerprint']!=_fingerprint(A,b,c,G,lo,hi):
        raise ValueError('Precomputed geometry belongs to a different instance.')
    fresh_witnesses=warm_data is None
    if fresh_witnesses:
        warm_data=corner_witnesses(A,b,c,G,lo,hi)
    warm_residual=_validate_witnesses(warm_data,A,b,c,G,lo,hi)
    errors=np.sum((np.asarray(warm_data['responses'])-anchor)**2,axis=1)
    index=int(np.argmax(errors))
    best_v=np.asarray(warm_data['parameters'][index])
    best_y=np.asarray(warm_data['responses'][index])
    best_lam=np.asarray(warm_data['multipliers'][index])
    incumbent=float(errors[index])
    box_upper=float(np.sum(np.maximum((geometry['y_lower']-anchor)**2,
                                      (geometry['y_upper']-anchor)**2)))
    build_begin=time.perf_counter()
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0)
        env.start()
        with gp.Model('global_projection_risk',env=env) as model:
            model.Params.OutputFlag=0
            model.Params.Threads=1
            model.Params.Seed=int(seed)
            model.Params.TimeLimit=float(time_limit)
            model.Params.NonConvex=2
            model.Params.MIPGap=float(relative_gap)
            model.Params.MIPGapAbs=float(absolute_gap)
            model.Params.FeasibilityTol=1e-8
            model.Params.IntFeasTol=1e-8
            model.Params.OptimalityTol=1e-8
            if log_path:
                model.Params.LogFile=str(log_path)
                model.Params.LogToConsole=0
                model.Params.OutputFlag=1
            v=[model.addVar(lb=float(lo[j]),ub=float(hi[j]),name=f'v{j}') for j in range(p)]
            y=[model.addVar(lb=float(geometry['y_lower'][j]),ub=float(geometry['y_upper'][j]),name=f'y{j}') for j in range(q)]
            lam=[model.addVar(lb=0.,ub=float(geometry['lambda_upper'][i]),name=f'lambda{i}')
                 if geometry['lambda_upper'][i]>0 else 0. for i in range(m)]
            active=[model.addVar(vtype=gp.GRB.BINARY,name=f'a{i}')
                    if geometry['lambda_upper'][i]>0 else None for i in range(m)]
            for i in range(m):
                row=gp.quicksum(float(A[i,j])*y[j] for j in range(q) if A[i,j])
                model.addConstr(row<=float(b[i]),name=f'primal{i}')
                if active[i] is None:
                    continue
                if complementarity=='bigm':
                    model.addConstr(lam[i]<=float(geometry['lambda_upper'][i])*active[i])
                    model.addConstr(float(b[i])-row<=float(geometry['slack_upper'][i])*(1-active[i]))
                elif complementarity=='indicator':
                    model.addGenConstrIndicator(active[i],False,lam[i]==0.)
                    model.addGenConstrIndicator(active[i],True,row==float(b[i]))
                else:
                    raise ValueError('Use complementarity=bigm or indicator.')
            for j in range(q):
                model.addConstr(y[j]-float(c[j])-gp.quicksum(float(G[j,k])*v[k] for k in range(p))
                                +gp.quicksum(float(A[i,j])*lam[i] for i in range(m) if A[i,j])==0,
                                name=f'stationarity{j}')
            coupled_capacity_upper=None
            if (capacity_structure_cut and m>=2*q
                    and np.array_equal(A[:q],-np.eye(q)) and np.array_equal(A[q:2*q],np.eye(q))
                    and np.array_equal(b[:q],np.zeros(q)) and np.all(b[q:]>0)
                    and np.all(A[2*q:]>=0) and np.array_equal(lo,-np.ones(p))
                    and np.array_equal(hi,np.ones(p))):
                from capacity_dual_bounds import capacity_multiplier_bounds
                structural=capacity_multiplier_bounds(A,b,c,G)
                coupled_capacity_upper=float(structural['stationarity_sum_upper'])
                # Multiplying stationarity by y and using complementarity
                # gives b.T lambda = z.T y - ||y||^2. Since 0<=y<=u,
                # this is at most max_v sum_j max_[0,u_j](z_j*t-t^2).
                # That convex function of v attains its box maximum at
                # a corner; structural['stationarity_sum_upper'] is the
                # padded maximum. This couples all multipliers, rather
                # than merely bounding each multiplier separately.
                model.addConstr(gp.quicksum(float(b[i])*lam[i] for i in range(m) if b[i])
                                <=coupled_capacity_upper,name='coupled_capacity_multiplier_bound')
            secant_error=0.
            if objective_form=='squared':
                objective=gp.quicksum((y[j]-float(anchor[j]))*(y[j]-float(anchor[j])) for j in range(q))
            elif objective_form=='stationarity':
                objective=(float(anchor@anchor)+gp.quicksum(float(c[j]-2*anchor[j])*y[j] for j in range(q))
                           +gp.quicksum(float(G[j,k])*y[j]*v[k] for j in range(q) for k in range(p) if G[j,k])
                           -gp.quicksum(float(b[i])*lam[i] for i in range(m)))
            elif objective_form=='pwl_secant':
                if int(pwl_segments)!=pwl_segments or pwl_segments<1:
                    raise ValueError('pwl_segments must be a positive integer.')
                objective=None
            else:
                raise ValueError('Use objective_form=squared, stationarity or pwl_secant.')
            if objective is not None:
                model.setObjective(objective,gp.GRB.MAXIMIZE)
            else:
                model.setObjective(0.,gp.GRB.MAXIMIZE)
                for j in range(q):
                    left=float(geometry['y_lower'][j]);right=float(geometry['y_upper'][j])
                    count=int(pwl_segments) if right-left>1e-6 else 1
                    grid=np.linspace(left,right,count+1)
                    values=(grid-float(anchor[j]))**2
                    model.setPWLObj(y[j],grid.tolist(),values.tolist())
                    secant_error+=float(np.max(np.diff(grid)**2))/4
            for j in range(p):v[j].Start=float(best_v[j])
            for j in range(q):y[j].Start=float(best_y[j])
            for i in range(m):
                if active[i] is not None:
                    lam[i].Start=float(best_lam[i])
                    active[i].Start=int(best_lam[i]>1e-10)
            model.update()
            variables=model.NumVars
            quadratic_nonzeros=model.NumQNZs
            build_seconds=time.perf_counter()-build_begin
            def terminate_on_bound(current,where):
                if target_upper is not None and where==gp.GRB.Callback.MIP:
                    bound=current.cbGet(gp.GRB.Callback.MIP_OBJBND)
                    if math.isfinite(bound) and bound+upper_backoff*max(1.,abs(bound))<target_upper:
                        current.terminate()
            model.optimize(terminate_on_bound if target_upper is not None else None)
            status=int(model.Status)
            solver_seconds=float(model.Runtime)
            nodes=float(model.NodeCount)
            sol_count=int(model.SolCount)
            raw_upper=float(model.ObjBound) if status not in [gp.GRB.INFEASIBLE,gp.GRB.INF_OR_UNBD,gp.GRB.UNBOUNDED] else math.inf
            raw_incumbent=float(model.ObjVal) if sol_count else None
            solver_v=np.array([z.X for z in v]) if sol_count else None
            solver_y=np.array([z.X for z in y]) if sol_count else None
            solver_lam=np.array([z.X if not isinstance(z,float) else z for z in lam]) if sol_count else None
            model_gap=float(model.MIPGap) if sol_count else math.inf
    validate_begin=time.perf_counter()
    validation_residual=0.;projection_difference=0.;objective_difference=0.
    checked_projection_residual=0.;parameter_clipping=0.
    if sol_count:
        projector=Projector(A,b,q)
        checked_v=np.clip(solver_v,lo,hi)
        parameter_clipping=float(np.max(np.abs(checked_v-solver_v),initial=0.))
        checked_y,_=projector(c+G@checked_v)
        checked_projection_residual=float(projector.maximum_kkt_residual)
        if checked_projection_residual>2e-7:
            raise RuntimeError('Independent incumbent projection failed KKT acceptance.')
        checked_risk=float(np.sum((checked_y-anchor)**2))
        slack=b-A@solver_y
        validation_residual=max(float(np.max(A@solver_y-b,initial=0.)),
                                float(np.max(-solver_lam,initial=0.)),
                                float(np.linalg.norm(solver_y-c-G@solver_v+A.T@solver_lam,np.inf)),
                                float(np.max(np.abs(solver_lam*slack),initial=0.)),
                                float(np.max(lo-solver_v,initial=0.)),float(np.max(solver_v-hi,initial=0.)),
                                projector.maximum_kkt_residual)
        projection_difference=float(np.linalg.norm(solver_y-checked_y,np.inf))
        objective_difference=abs(raw_incumbent-checked_risk)
        if checked_risk>incumbent:
            incumbent=checked_risk;best_v=checked_v;best_y=checked_y
    upper=box_upper
    if math.isfinite(raw_upper) and status!=gp.GRB.NUMERIC:
        upper=min(upper,raw_upper+upper_backoff*max(1.,abs(raw_upper)))
    if upper<incumbent-2e-6*max(1.,abs(incumbent)):
        raise RuntimeError('Global upper risk falls below an independently feasible response.')
    upper=max(upper,incumbent)
    status_names={getattr(gp.GRB,n):n.lower() for n in ['OPTIMAL','TIME_LIMIT','NODE_LIMIT','SOLUTION_LIMIT','SUBOPTIMAL','NUMERIC','INTERRUPTED','INFEASIBLE','INF_OR_UNBD','UNBOUNDED']}
    return dict(upper=float(upper),incumbent=float(incumbent),lower=float(incumbent),
                raw_solver_upper=raw_upper,raw_solver_incumbent=raw_incumbent,box_upper=box_upper,
                status=status_names.get(status,str(status)),status_code=status,gap=model_gap,
                absolute_gap=float(upper-incumbent),variables=variables,nodes=nodes,solutions=sol_count,
                parameter=best_v.tolist(),response=best_y.tolist(),
                maximum_kkt_residual=max(float(validation_residual),warm_residual),projection_difference=projection_difference,
                objective_difference=objective_difference,solver_seconds=solver_seconds,
                independent_projection_residual=checked_projection_residual,
                parameter_clipping_inf_norm=parameter_clipping,warm_witness_kkt_residual=warm_residual,
                build_seconds=build_seconds,validation_seconds=time.perf_counter()-validate_begin,
                geometry_seconds=float(geometry['preparation_seconds']) if fresh_geometry else 0.,
                witness_seconds=float(warm_data['seconds']) if fresh_witnesses else 0.,
                total_seconds=time.perf_counter()-started,solver='Gurobi',solver_version=list(gp.gurobi.version()),
                complementarity=complementarity,objective_form=objective_form,
                slater_margin=float(geometry['slater_margin']),
                maximum_dual_bound=float(np.max(geometry['lambda_upper'])),
                exact_zero_multipliers=len(geometry.get('exact_zero_multipliers',[])),
                capacity_structure_cut_added=coupled_capacity_upper is not None,
                capacity_stationarity_upper=coupled_capacity_upper,
                quadratic_objective_nonzeros=quadratic_nonzeros,
                secant_error_bound=secant_error,
                pwl_segments=int(pwl_segments) if objective_form=='pwl_secant' else None,
                upper_backoff=float(upper_backoff),threads=1,seed=int(seed))


def solve_pwl_global_upper(A,b,c,G,anchor,time_limit=60.,*,segments=32,**kwargs):
    """A MILP upper approximation with exact projection KKT and secant objective.

    For each bounded action coordinate the quadratic is replaced by its
    linear interpolant on a uniform grid. Convexity makes this an upper
    function; the total overestimation is at most sum(max_step_j^2)/4.
    Therefore ObjBound is directly an upper risk, while ObjVal is *not*
    a feasible lower risk. The returned lower risk is freshly projected.
    """
    return solve_global_upper(A,b,c,G,anchor,time_limit=time_limit,
                              objective_form='pwl_secant',pwl_segments=segments,**kwargs)


def solve_box_global_upper(A,b,c,G,anchor,time_limit=60.,*,geometry=None,warm_data=None,
                           v_lower=None,v_upper=None,complementarity='bigm',seed=271127,
                           log_path=None,relative_gap=1e-6,absolute_gap=1e-7,
                           upper_backoff=1e-7,target_upper=None):
    """Exact compact KKT model when A=[-I; I; C] and b=[0; u; d].

    For shared-constraint multipliers eta, box stationarity is equivalent to
    y=clip(c+Gv-C.T eta,0,u). Exact three-piece PWL constraints eliminate
    all box multipliers and their disjunctions from the input model.
    Complementarity remains only for the shared C rows. The same Slater
    argument bounds eta, so no unproved truncation is used.
    """
    started=time.perf_counter()
    A,b,c,G,anchor=map(lambda x:np.asarray(x,dtype=float),(A,b,c,G,anchor))
    m,q=A.shape;p=G.shape[1]
    if m<2*q or not np.allclose(A[:q],-np.eye(q),atol=0,rtol=0) or not np.allclose(A[q:2*q],np.eye(q),atol=0,rtol=0) or not np.allclose(b[:q],0,atol=0,rtol=0):
        raise ValueError('Compact form requires exact rows A=[-I;I;C], b=[0;u;d].')
    capacity=A[2*q:];rhs=b[2*q:];u=b[q:2*q];k=len(rhs)
    if np.any(u<=0):
        raise ValueError('Positive box upper bounds required.')
    lo=np.full(p,-1.) if v_lower is None else np.asarray(v_lower,dtype=float)
    hi=np.full(p,1.) if v_upper is None else np.asarray(v_upper,dtype=float)
    fresh_geometry=geometry is None
    if fresh_geometry:
        geometry=derive_projection_bounds(A,b,c,G,lo,hi)
    if geometry['fingerprint']!=_fingerprint(A,b,c,G,lo,hi):
        raise ValueError('Precomputed geometry belongs to a different instance.')
    fresh_witnesses=warm_data is None
    if fresh_witnesses:
        warm_data=corner_witnesses(A,b,c,G,lo,hi)
    warm_residual=_validate_witnesses(warm_data,A,b,c,G,lo,hi)
    errors=np.sum((np.asarray(warm_data['responses'])-anchor)**2,axis=1)
    index=int(np.argmax(errors));incumbent=float(errors[index])
    best_v=np.asarray(warm_data['parameters'][index]);best_y=np.asarray(warm_data['responses'][index])
    initial_eta=np.asarray(warm_data['multipliers'][index])[2*q:]
    eta_bound=geometry['lambda_upper'][2*q:]
    preferred_min=c+np.maximum(G,0)@lo+np.minimum(G,0)@hi
    preferred_max=c+np.maximum(G,0)@hi+np.minimum(G,0)@lo
    w_lower=preferred_min-np.maximum(capacity,0).T@eta_bound
    w_upper=preferred_max-np.minimum(capacity,0).T@eta_bound
    box_upper=float(np.sum(np.maximum((geometry['y_lower']-anchor)**2,(geometry['y_upper']-anchor)**2)))
    build_begin=time.perf_counter()
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        with gp.Model('global_box_projection_risk',env=env) as model:
            for name,value in [('OutputFlag',0),('Threads',1),('Seed',int(seed)),('TimeLimit',float(time_limit)),
                               ('NonConvex',2),('MIPGap',float(relative_gap)),('MIPGapAbs',float(absolute_gap)),
                               ('FeasibilityTol',1e-8),('IntFeasTol',1e-8),('OptimalityTol',1e-8)]:
                model.setParam(name,value)
            if log_path:
                model.Params.LogFile=str(log_path);model.Params.LogToConsole=0;model.Params.OutputFlag=1
            v=[model.addVar(lb=float(lo[j]),ub=float(hi[j]),name=f'v{j}') for j in range(p)]
            y=[model.addVar(lb=float(geometry['y_lower'][j]),ub=float(geometry['y_upper'][j]),name=f'y{j}') for j in range(q)]
            w=[model.addVar(lb=float(w_lower[j]),ub=float(w_upper[j]),name=f'w{j}') for j in range(q)]
            eta=[model.addVar(lb=0.,ub=float(eta_bound[i]),name=f'eta{i}') if eta_bound[i]>0 else 0. for i in range(k)]
            active=[model.addVar(vtype=gp.GRB.BINARY,name=f'a{i}') if eta_bound[i]>0 else None for i in range(k)]
            for j in range(q):
                model.addConstr(w[j]==float(c[j])+gp.quicksum(float(G[j,l])*v[l] for l in range(p))
                                -gp.quicksum(float(capacity[i,j])*eta[i] for i in range(k) if capacity[i,j]))
                xpoints=[float(min(-1.,w_lower[j]-1.)),0.,float(u[j]),float(max(u[j]+1.,w_upper[j]+1.))]
                model.addGenConstrPWL(w[j],y[j],xpoints,[0.,0.,float(u[j]),float(u[j])],name=f'clip{j}')
            for i in range(k):
                row=gp.quicksum(float(capacity[i,j])*y[j] for j in range(q) if capacity[i,j])
                model.addConstr(row<=float(rhs[i]))
                if active[i] is None:
                    continue
                if complementarity=='bigm':
                    model.addConstr(eta[i]<=float(eta_bound[i])*active[i])
                    model.addConstr(float(rhs[i])-row<=float(geometry['slack_upper'][2*q+i])*(1-active[i]))
                elif complementarity=='indicator':
                    model.addGenConstrIndicator(active[i],False,eta[i]==0.)
                    model.addGenConstrIndicator(active[i],True,row==float(rhs[i]))
                else:
                    raise ValueError('Use complementarity=bigm or indicator.')
            model.setObjective(gp.quicksum((y[j]-float(anchor[j]))*(y[j]-float(anchor[j])) for j in range(q)),gp.GRB.MAXIMIZE)
            warm_w=c+G@best_v-capacity.T@initial_eta
            for j in range(p):v[j].Start=float(best_v[j])
            for j in range(q):y[j].Start=float(best_y[j]);w[j].Start=float(warm_w[j])
            for i in range(k):
                if active[i] is not None:
                    eta[i].Start=float(initial_eta[i]);active[i].Start=int(initial_eta[i]>1e-10)
            model.update();variables=model.NumVars
            build_seconds=time.perf_counter()-build_begin
            def terminate_on_bound(current,where):
                if target_upper is not None and where==gp.GRB.Callback.MIP:
                    bound=current.cbGet(gp.GRB.Callback.MIP_OBJBND)
                    if math.isfinite(bound) and bound+upper_backoff*max(1.,abs(bound))<target_upper:
                        current.terminate()
            model.optimize(terminate_on_bound if target_upper is not None else None)
            status=int(model.Status);solver_seconds=float(model.Runtime);nodes=float(model.NodeCount)
            count=int(model.SolCount)
            raw_upper=float(model.ObjBound) if status not in [gp.GRB.INFEASIBLE,gp.GRB.INF_OR_UNBD,gp.GRB.UNBOUNDED] else math.inf
            raw_incumbent=float(model.ObjVal) if count else None
            solver_v=np.array([a.X for a in v]) if count else None
            solver_y=np.array([a.X for a in y]) if count else None
            solver_w=np.array([a.X for a in w]) if count else None
            solver_eta=np.array([a.X if not isinstance(a,float) else a for a in eta]) if count else None
            gap=float(model.MIPGap) if count else math.inf
    check_begin=time.perf_counter();residual=projection_difference=objective_difference=0.
    checked_projection_residual=0.;parameter_clipping=0.
    if count:
        projector=Projector(A,b,q);checked_v=np.clip(solver_v,lo,hi)
        parameter_clipping=float(np.max(np.abs(checked_v-solver_v),initial=0.))
        checked_y,_=projector(c+G@checked_v)
        checked_projection_residual=float(projector.maximum_kkt_residual)
        if checked_projection_residual>2e-7:
            raise RuntimeError('Independent incumbent projection failed KKT acceptance.')
        checked_risk=float(np.sum((checked_y-anchor)**2))
        lam=np.r_[np.maximum(-solver_w,0),np.maximum(solver_w-u,0),solver_eta]
        residual=max(float(np.max(A@solver_y-b,initial=0.)),float(np.max(-lam,initial=0.)),
                     float(np.linalg.norm(solver_y-c-G@solver_v+A.T@lam,np.inf)),
                     float(np.max(np.abs(lam*(b-A@solver_y)),initial=0.)),
                     float(np.max(lo-solver_v,initial=0.)),float(np.max(solver_v-hi,initial=0.)),
                     projector.maximum_kkt_residual)
        projection_difference=float(np.linalg.norm(solver_y-checked_y,np.inf))
        objective_difference=abs(raw_incumbent-checked_risk)
        if checked_risk>incumbent:
            incumbent=checked_risk;best_v=checked_v;best_y=checked_y
    upper=box_upper
    if math.isfinite(raw_upper) and status!=gp.GRB.NUMERIC:
        upper=min(upper,raw_upper+upper_backoff*max(1.,abs(raw_upper)))
    if upper<incumbent-2e-6*max(1.,abs(incumbent)):
        raise RuntimeError('Global upper risk falls below independently feasible risk.')
    upper=max(upper,incumbent)
    names={getattr(gp.GRB,n):n.lower() for n in ['OPTIMAL','TIME_LIMIT','NODE_LIMIT','SOLUTION_LIMIT','SUBOPTIMAL','NUMERIC','INTERRUPTED','INFEASIBLE','INF_OR_UNBD','UNBOUNDED']}
    return dict(upper=float(upper),incumbent=float(incumbent),lower=float(incumbent),raw_solver_upper=raw_upper,
                raw_solver_incumbent=raw_incumbent,box_upper=box_upper,status=names.get(status,str(status)),status_code=status,
                gap=gap,absolute_gap=float(upper-incumbent),variables=variables,nodes=nodes,solutions=count,
                parameter=best_v.tolist(),response=best_y.tolist(),maximum_kkt_residual=max(float(residual),warm_residual),
                projection_difference=projection_difference,objective_difference=objective_difference,
                independent_projection_residual=checked_projection_residual,
                parameter_clipping_inf_norm=parameter_clipping,warm_witness_kkt_residual=warm_residual,
                solver_seconds=solver_seconds,build_seconds=build_seconds,validation_seconds=time.perf_counter()-check_begin,
                geometry_seconds=float(geometry['preparation_seconds']) if fresh_geometry else 0.,
                witness_seconds=float(warm_data['seconds']) if fresh_witnesses else 0.,total_seconds=time.perf_counter()-started,
                solver='Gurobi',solver_version=list(gp.gurobi.version()),complementarity=complementarity,
                objective_form='squared',formulation='box_partial_dual_pwl',slater_margin=float(geometry['slater_margin']),
                maximum_dual_bound=float(np.max(geometry['lambda_upper'])),upper_backoff=float(upper_backoff),threads=1,seed=int(seed))


if __name__=='__main__':
    # Small exact projection control; the farthest point is (1,1).
    A=np.r_[-np.eye(2),np.eye(2)]
    b=np.r_[np.zeros(2),np.ones(2)]
    result=solve_global_upper(A,b,np.zeros(2),np.eye(2),np.zeros(2),time_limit=10.)
    print(json.dumps(result,indent=2))
