"""Radius-adaptive lower certificates for affine projection images.

Target convention: A x<=b and E x=f. Concrete distance solvers support a
scalar interval, or a parameter ball with homogeneous conic source fitting.
The search also accepts a user-supplied distance oracle for other convex
inverse classes. Floating-point lower values are diagnostics, not intervals.
"""
import numpy as np
import cvxpy as cp


def cone_distance(latent,offset,normals,anchor,*,interval=None,radius=None,
                  equalities=None,sources=(),budget=0.):
    """Primal/dual distance to anchor+cone(normals.T)+range(E.T).

For a ball prior, each source is (H_i,B_i), C_i={u:H_i u>=0}, and fitting
means sum ||proj(C_i,B_i v)||^2<=2*N*budget. An interval prior has p=1.
"""
    G=np.asarray(latent,float);G=G[:,None] if G.ndim==1 else G
    c=np.asarray(offset,float);x=np.asarray(anchor,float);q,p=G.shape
    A=np.asarray(normals,float).reshape((-1,q))
    E=np.empty((0,q)) if equalities is None else np.asarray(equalities,float).reshape((-1,q))
    v=cp.Variable(p);normal=cp.Constant(np.zeros(q));constraints=[]
    if len(A):normal=normal+A.T@cp.Variable(len(A),nonneg=True)
    if len(E):normal=normal+E.T@cp.Variable(len(E))
    if interval is not None:
        if p!=1 or sources:raise ValueError('Interval prior requires p=1 and no additional source constraints')
        left,right=map(float,interval)
        if not np.isfinite([left,right]).all() or left>right:raise ValueError('Invalid interval')
        constraints.extend([v[0]>=left,v[0]<=right])
    else:
        if radius is None or radius<=0 or budget<0:raise ValueError('Positive ball radius and nonnegative budget required')
        constraints.append(cp.norm(v)<=radius)
    source_data=[];errors=[]
    for rows,map_ in sources:
        B=np.asarray(map_,float);H=np.asarray(rows,float).reshape((-1,B.shape[0]))
        error=B@v
        if len(H):error=error+H.T@cp.Variable(len(H),nonneg=True)
        errors.append(error);source_data.append((H,B))
    if errors:
        residual=cp.hstack(errors)
        constraints.append(residual==0 if budget==0 else cp.norm(residual)<=np.sqrt(2*len(errors)*budget))
    primal=cp.Problem(cp.Minimize(cp.norm(c+G@v-x-normal)),constraints)
    settings=dict(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=400)
    primal.solve(**settings)
    w=cp.Variable(q);dual_constraints=[cp.norm(w)<=1.]
    if len(A):dual_constraints.append(A@w>=0)
    if len(E):dual_constraints.append(E@w==0)
    support_vector=G.T@w;dual_sources=[]
    if interval is not None:
        support=cp.Variable()
        dual_constraints.extend([support>=left*support_vector[0],support>=right*support_vector[0]])
    else:
        for H,B in source_data:
            variable=cp.Variable(B.shape[0]);dual_sources.append(variable)
            if len(H):dual_constraints.append(H@variable>=0)
            support_vector=support_vector-B.T@variable
        support=radius*cp.norm(support_vector)
        if dual_sources and budget>0:support+=np.sqrt(2*len(dual_sources)*budget)*cp.norm(cp.hstack(dual_sources))
    dual=cp.Problem(cp.Maximize((x-c)@w-support),dual_constraints)
    dual.solve(**settings)
    if primal.status not in ['optimal','optimal_inaccurate'] or dual.status not in ['optimal','optimal_inaccurate']:
        raise RuntimeError('Cone-distance solve failed: '+str((primal.status,dual.status)))
    W=np.asarray(w.value).ravel()
    violation=max(0.,np.linalg.norm(W)-1.,
        float(np.max(-A@W)) if len(A) else 0.,float(np.max(np.abs(E@W))) if len(E) else 0.)
    for (H,_),variable in zip(source_data,dual_sources):
        if len(H):violation=max(violation,float(np.max(-H@variable.value)))
    if violation>1e-7:raise RuntimeError('Dual feasibility residual too large')
    return dict(upper=float(primal.value),lower=max(0.,float(dual.value)-1e-7),
                primal_status=primal.status,dual_status=dual.status,
                dual_feasibility_residual=violation,numerical_backoff=1e-7)


def radius_search(rows,bounds,anchor,distance_oracle,*,activity_tolerance=2e-7,
                  exhaustive=False):
    """Maximize min(rho,Delta_rho) over the finite slack candidates.

distance_oracle(selected_rows) returns a dict with lower/upper distance values.
The exact-arithmetic solve count counts candidate distances, not primal and
dual solver invocations separately. Numerical search always reports a valid
candidate lower formula conditional on feasibility of its numerical dual.
"""
    x=np.asarray(anchor,float);A=np.asarray(rows,float).reshape((-1,len(x)))
    b=np.asarray(bounds,float);slack=b-A@x
    if len(slack) and slack.min()<-activity_tolerance:raise ValueError('Target anchor infeasible')
    norms=np.linalg.norm(A,axis=1);valid=norms>0
    A=A[valid];slack=slack[valid];norms=norms[valid]
    distances=np.maximum(slack,0)/norms;active=distances<activity_tolerance
    thresholds=sorted(set(float(s) for s in distances[~active]))+[np.inf]
    cache={}
    def evaluate(i):
        if i not in cache:
            rho=thresholds[i];selected=active|(distances<rho)
            value=distance_oracle(A[selected])
            cache[i]=dict(index=i,radius=rho,upper=value['upper'],lower=value['lower'],
                          image_distance_lower=min(rho,value['lower']),rows=int(selected.sum()))
        return cache[i]
    if exhaustive:
        for i in range(len(thresholds)):evaluate(i)
    else:
        lo,hi=0,len(thresholds)-1
        while lo<hi:
            mid=(lo+hi)//2;value=evaluate(mid)
            if value['upper']<=value['radius']:hi=mid
            else:lo=mid+1
        evaluate(lo)
        if lo:evaluate(lo-1)
    best=max(cache.values(),key=lambda r:r['image_distance_lower'])
    return dict(lower=best['image_distance_lower'],best=best,solves=len(cache),
                thresholds=len(thresholds),evaluations=list(cache.values()))


def adaptive_certificate(latent,offset,rows,bounds,anchor,*,interval=None,radius=None,
                         equalities=None,sources=(),budget=0.,exhaustive=False):
    oracle=lambda normals:cone_distance(latent,offset,normals,anchor,interval=interval,
        radius=radius,equalities=equalities,sources=sources,budget=budget)
    return radius_search(rows,bounds,anchor,oracle,exhaustive=exhaustive)


def original_interval_certificate(rows,bounds,offset,direction,interval,anchor):
    """Original single-margin lower formula, for the matched numerical baseline."""
    A=np.asarray(rows,float);b=np.asarray(bounds,float);x=np.asarray(anchor,float)
    slack=b-A@x;active=slack<2e-7;inactive=~active
    alpha=float(np.min(slack[inactive]/np.linalg.norm(A[inactive],axis=1))) if inactive.any() else np.inf
    result=cone_distance(direction,offset,A[active],x,interval=interval)
    return result['upper'],result['lower'],alpha
