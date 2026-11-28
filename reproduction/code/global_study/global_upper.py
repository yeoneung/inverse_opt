"""Global upper risks for an affine family of polyhedral Euclidean projections.

The model is an exact KKT/SOS1 reformulation in exact arithmetic. Reported
SCIP bounds, feasibility checks and padding use floating-point arithmetic;
they are not interval certificates. A maximization INCUMBENT is a lower
bound on the worst risk. Only the global dual bound is an upper risk.
"""
from itertools import product
from pathlib import Path
import sys
import time

import clarabel
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

VENDOR=Path(__file__).resolve().parent/'vendor'
if VENDOR.is_dir():
    sys.path.insert(0,str(VENDOR))
from pyscipopt import Model, quicksum


class ForwardProjector:
    """Independent convex QP solves used for witnesses and SCIP validation."""
    def __init__(self,A,b):
        self.A=np.asarray(A,float)
        self.b=np.asarray(b,float)
        q=self.A.shape[1]
        settings=clarabel.DefaultSettings()
        settings.verbose=False
        settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-11
        settings.max_iter=500
        self.solver=clarabel.DefaultSolver(sparse.eye(q,format='csc'),np.zeros(q),
            sparse.csc_matrix(self.A),self.b,[clarabel.NonnegativeConeT(len(self.b))],settings)
        self.calls=0

    def __call__(self,z):
        z=np.asarray(z,float)
        self.solver.update(q=-z)
        result=self.solver.solve()
        if str(result.status) not in ['Solved','AlmostSolved']:
            raise RuntimeError('Independent projection failed: '+str(result.status))
        y=np.asarray(result.x)
        lam=np.asarray(result.z)
        # Recover a basic active multiplier representation when it passes KKT.
        slack=self.b-self.A@y
        if np.max(self.A@z-self.b,initial=0.)<=0:
            y=z.copy();lam=np.zeros(len(self.b))
        else:
            choices=[np.flatnonzero(lam>=np.maximum(slack,0.))]
            choices += [np.flatnonzero(slack<tol) for tol in [1e-5,1e-6,1e-7,1e-8]]
            for active in choices:
                B=self.A[active]
                if not len(active) or np.linalg.matrix_rank(B)!=len(active):
                    continue
                multipliers=np.linalg.solve(B@B.T,B@z-self.b[active])
                candidate=z-B.T@multipliers
                if multipliers.min()>=-1e-10 and np.max(self.A@candidate-self.b,initial=0.)<=1e-10:
                    y=candidate;lam=np.zeros(len(self.b));lam[active]=multipliers
                    break
        self.calls+=1
        residual=projection_kkt_residual(self.A,self.b,z,y,lam)
        if max(residual.values())>2e-7:
            raise RuntimeError('Independent projection failed its KKT residual check')
        return y,lam


def box_corner_parameters(p):
    return np.asarray(list(product([-1.,1.],repeat=p)),float)


def projection_kkt_residual(A,b,z,y,lam):
    slack=np.asarray(b)-np.asarray(A)@y
    return dict(primal=float(max(0.,np.max(-slack,initial=0.))),
        dual=float(max(0.,np.max(-lam,initial=0.))),
        stationarity=float(np.max(np.abs(y-z+np.asarray(A).T@lam),initial=0.)),
        complementarity=float(np.max(np.abs(lam*slack),initial=0.)))


def derive_projection_bounds(A,b,c,G,*,margin=1e-9):
    """Derive action and optional dual bounds without guessed constants.

    If a strict Slater point ybar exists, all projection multipliers obey
    lambda_i <= C/(b_i-A_i*ybar), where C bounds .5||ybar-c-Gv||^2 uniformly.
    Coordinate LPs are unnecessary when explicit box rows already give bounds.
    """
    start=time.perf_counter()
    A=np.asarray(A,float);b=np.asarray(b,float);c=np.asarray(c,float);G=np.asarray(G,float)
    m,q=A.shape;p=G.shape[1]
    assert b.shape==(m,) and c.shape==(q,) and G.shape[0]==q
    lower=np.full(q,-np.inf);upper=np.full(q,np.inf)
    for row,rhs in zip(A,b):
        nonzero=np.flatnonzero(row)
        if len(nonzero)==1:
            j=nonzero[0]
            if row[j]>0: upper[j]=min(upper[j],rhs/row[j])
            else: lower[j]=max(lower[j],rhs/row[j])
        elif not len(nonzero) and rhs<0:
            raise ValueError('Infeasible constant target row')
    lp_count=0
    for j in range(q):
        direction=np.zeros(q);direction[j]=1.
        for sense,bounds in [(1.,lower),(-1.,upper)]:
            if np.isfinite(bounds[j]): continue
            result=linprog(sense*direction,A_ub=A,b_ub=b,bounds=[(None,None)]*q,method='highs')
            lp_count+=1
            if result.success:
                bounds[j]=sense*result.fun
            elif result.status!=3:
                raise RuntimeError('Target action-bound LP: '+result.message)
    # An unbounded target can still have bounded responses over a compact prior.
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        projector=ForwardProjector(A,b)
        center,_=projector(c)
        radius=np.linalg.norm(G,2)*np.sqrt(p)
        lower=np.maximum(lower,center-radius)
        upper=np.minimum(upper,center+radius)
    if np.any(lower>upper+1e-7):
        raise RuntimeError('Inconsistent action bounds')
    padding=margin*np.maximum(1.,np.maximum(np.abs(lower),np.abs(upper)))
    lower=lower-padding;upper=upper+padding
    rownorm=np.linalg.norm(A,axis=1)
    nonzero=rownorm>0
    slater=linprog(np.r_[np.zeros(q),-1.],
        A_ub=np.column_stack([A,rownorm]),b_ub=b,
        bounds=[(None,None)]*q+[(0.,None)],method='highs')
    lp_count+=1
    if slater.status==3:
        # Unbounded inradius: any feasible point with radius one suffices.
        slater=linprog(np.zeros(q),A_ub=A,b_ub=b-rownorm,
            bounds=[(None,None)]*q,method='highs')
        lp_count+=1
    point=slater.x[:q] if slater.success else None
    slacks=b-A@point if point is not None else None
    dual_upper=np.full(m,np.inf)
    dual_upper[~nonzero]=0.
    objective_gap=None
    if point is not None and np.all(slacks[nonzero]>1e-10):
        if p<=12:
            parameters=box_corner_parameters(p)
            differences=point-c-parameters@G.T
            objective_gap=.5*float(np.max(np.sum(differences*differences,axis=1)))
            gap_method='convex_quadratic_at_all_box_corners'
        else:
            objective_gap=.5*float(np.sum((np.abs(point-c)+np.sum(np.abs(G),axis=1))**2))
            gap_method='componentwise_uniform_bound'
        dual_upper[nonzero]=objective_gap/slacks[nonzero]
        dual_upper[nonzero]+=margin*np.maximum(1.,dual_upper[nonzero])
    else:
        gap_method='strict_slater_bound_unavailable;SOS1_requires_no_big_M'
    slack_upper=b-A.clip(max=0)@upper-A.clip(min=0)@lower
    slack_upper=np.maximum(slack_upper,0.)+margin*np.maximum(1.,np.abs(slack_upper))
    return dict(y_lower=lower,y_upper=upper,lambda_upper=dual_upper,
        slack_upper=slack_upper,slater_point=point,slater_slack=slacks,
        uniform_objective_gap_upper=objective_gap,dual_bound_method=gap_method,
        linear_programs=lp_count,seconds=time.perf_counter()-start)


def corner_witnesses(A,b,c,G,*,projector=None,include_center=True):
    """Forward responses; these are feasible incumbents, never global upper bounds."""
    p=np.asarray(G).shape[1]
    projector=projector or ForwardProjector(A,b)
    parameters=box_corner_parameters(p)
    if include_center: parameters=np.vstack([np.zeros((1,p)),parameters])
    records=[]
    for v in parameters:
        y,lam=projector(np.asarray(c)+np.asarray(G)@v)
        records.append(dict(parameter=v,response=y,multiplier=lam))
    return records


def global_upper(A,b,c,G,anchor,*,time_limit=60.,relative_gap=1e-6,
                 absolute_gap=1e-7,feasibility_tolerance=1e-8,
                 bound_margin=1e-7,warmstart_parameters=None,
                 warmstart_records=None,bounds=None,verbose=False,seed=0):
    """Return a numerically globally bounded worst risk for a fixed anchor.

    total_wall_seconds includes preprocessing, witness QPs, model construction,
    SCIP optimization and independent validation. time_limit applies to SCIP.
    Precomputed bounds and witness records can be reused in anchor exchange.
    """
    start=time.perf_counter()
    A=np.asarray(A,float);b=np.asarray(b,float);c=np.asarray(c,float);G=np.asarray(G,float)
    anchor=np.asarray(anchor,float);m,q=A.shape;p=G.shape[1]
    preprocess=time.perf_counter()
    bounds=derive_projection_bounds(A,b,c,G) if bounds is None else bounds
    preprocessing_seconds=time.perf_counter()-preprocess
    lower,upper=bounds['y_lower'],bounds['y_upper']
    box_upper=float(np.sum(np.maximum((lower-anchor)**2,(upper-anchor)**2)))
    projector=ForwardProjector(A,b)
    begin=time.perf_counter()
    records=[] if warmstart_records is None else list(warmstart_records)
    if not records:
        if warmstart_parameters is None:
            warmstart_parameters=(np.vstack([np.zeros((1,p)),box_corner_parameters(p)])
                if p<=12 else np.vstack([np.zeros((1,p)),np.eye(p),-np.eye(p)]))
        for v in warmstart_parameters:
            v=np.asarray(v,float)
            assert np.max(np.abs(v),initial=0.)<=1+1e-12
            v=np.clip(v,-1.,1.)
            y,lam=projector(c+G@v)
            records.append(dict(parameter=v,response=y,multiplier=lam))
    warmstart_seconds=time.perf_counter()-begin
    records.sort(key=lambda row:float(np.sum((np.asarray(row['response'])-anchor)**2)),reverse=True)
    best=records[0]
    # Reproject the best supplied parameter to obtain an independently checked
    # lower value. Warmstart records need not be trusted as exact QP solutions.
    best_parameter=np.clip(np.asarray(best['parameter'],float),-1.,1.)
    best_response,best_multiplier=projector(c+G@best_parameter)
    best=dict(parameter=best_parameter,response=best_response,multiplier=best_multiplier)
    incumbent=float(np.sum((best_response-anchor)**2))
    warmstart_seconds=time.perf_counter()-begin
    begin=time.perf_counter()
    model=Model('projection_worst_risk')
    if not verbose: model.hideOutput()
    model.setIntParam('parallel/maxnthreads',1)
    model.setIntParam('lp/threads',1)
    model.setIntParam('randomization/randomseedshift',int(seed))
    model.setRealParam('limits/time',float(time_limit))
    model.setRealParam('limits/gap',float(relative_gap))
    model.setRealParam('limits/absgap',float(absolute_gap))
    model.setRealParam('numerics/feastol',float(feasibility_tolerance))
    v=[model.addVar(f'v{j}',lb=-1.,ub=1.) for j in range(p)]
    y=[model.addVar(f'y{j}',lb=float(lower[j]),ub=float(upper[j])) for j in range(q)]
    lam=[model.addVar(f'lambda{i}',lb=0.,ub=float(bounds['lambda_upper'][i])
          if np.isfinite(bounds['lambda_upper'][i]) else None) for i in range(m)]
    slack=[model.addVar(f'slack{i}',lb=0.,ub=float(bounds['slack_upper'][i])) for i in range(m)]
    for i in range(m):
        model.addCons(slack[i]+quicksum(float(A[i,j])*y[j] for j in range(q))==float(b[i]))
        model.addConsSOS1([lam[i],slack[i]],name=f'complementarity{i}')
    for j in range(q):
        model.addCons(y[j]-float(c[j])-quicksum(float(G[j,k])*v[k] for k in range(p))
                      +quicksum(float(A[i,j])*lam[i] for i in range(m))==0.)
    t=model.addVar('worst_risk',lb=0.,ub=box_upper)
    model.addCons(t<=quicksum((y[j]-float(anchor[j]))**2 for j in range(q)),name='risk_hypograph')
    model.setObjective(t,'maximize')
    accepted_warmstarts=0
    for record in records[:20]:
        sol=model.createSol()
        yy=np.asarray(record['response']);ll=np.asarray(record['multiplier']).copy()
        ll[np.linalg.norm(A,axis=1)==0]=0.
        ss=np.maximum(b-A@yy,0.)
        risk=float(np.sum((yy-anchor)**2))
        for variable,value in zip(v,record['parameter']): model.setSolVal(sol,variable,float(value))
        for variable,value in zip(y,yy): model.setSolVal(sol,variable,float(value))
        for variable,value in zip(lam,ll): model.setSolVal(sol,variable,float(max(value,0.)))
        for variable,value in zip(slack,ss): model.setSolVal(sol,variable,float(value))
        model.setSolVal(sol,t,max(0.,risk-1e-9*max(1.,risk)))
        accepted_warmstarts+=bool(model.addSol(sol,free=True))
    construction_seconds=time.perf_counter()-begin
    begin=time.perf_counter()
    model.optimize()
    optimize_seconds=time.perf_counter()-begin
    status=str(model.getStatus())
    solver_upper=float(model.getDualbound())
    solver_lower=float(model.getPrimalbound())
    begin=time.perf_counter()
    best_solution=model.getBestSol()
    validation={}
    if best_solution is not None:
        raw_v=np.asarray([model.getSolVal(best_solution,z) for z in v])
        vv=np.clip(raw_v,-1.,1.)
        yy=np.asarray([model.getSolVal(best_solution,z) for z in y])
        ll=np.asarray([model.getSolVal(best_solution,z) for z in lam])
        actual,actual_lam=projector(c+G@vv)
        actual_risk=float(np.sum((actual-anchor)**2))
        validation=projection_kkt_residual(A,b,c+G@vv,yy,ll)
        validation['independent_projection_error']=float(np.linalg.norm(yy-actual,np.inf))
        validation['independent_projection_kkt']=projection_kkt_residual(A,b,c+G@vv,actual,actual_lam)
        validation['returned_parameter_feasibility']=float(max(0.,np.max(np.abs(vv))-1.))
        validation['solver_parameter_feasibility']=float(max(0.,np.max(np.abs(raw_v))-1.))
        if actual_risk>incumbent:
            incumbent=actual_risk
            best=dict(parameter=vv,response=actual,multiplier=actual_lam)
    finite_upper=solver_upper if np.isfinite(solver_upper) and solver_upper<1e19 else box_upper
    padded_upper=min(box_upper,finite_upper+bound_margin*max(1.,abs(finite_upper)))
    if padded_upper<incumbent-5e-6*max(1.,incumbent):
        raise RuntimeError('Solver upper bound is below independently feasible risk')
    # The independently verified incumbent protects against tiny bound rounding.
    padded_upper=max(padded_upper,incumbent)
    validation_seconds=time.perf_counter()-begin
    result=dict(status=status,lower_bound=incumbent,upper_bound=padded_upper,
        solver_primal_bound=solver_lower,solver_dual_bound=solver_upper,
        relative_gap=float(model.getGap()),absolute_gap=max(0.,padded_upper-incumbent),
        worst_parameter=np.asarray(best['parameter']).tolist(),
        worst_response=np.asarray(best['response']).tolist(),
        solver_nodes=int(model.getNNodes()),scip_seconds=float(model.getSolvingTime()),
        preprocessing_seconds=preprocessing_seconds,warmstart_seconds=warmstart_seconds,
        model_construction_seconds=construction_seconds,optimization_wall_seconds=optimize_seconds,
        independent_validation_seconds=validation_seconds,total_wall_seconds=time.perf_counter()-start,
        warmstart_records=len(records),accepted_warmstarts=accepted_warmstarts,
        independent_forward_queries=projector.calls,kkt_validation=validation,
        action_box_upper=box_upper,dual_bound_method=bounds['dual_bound_method'],
        all_multiplier_bounds_finite=bool(np.all(np.isfinite(bounds['lambda_upper']))),
        numerical_bound_margin=bound_margin,interval_validated=False)
    model.freeProb()
    return result
