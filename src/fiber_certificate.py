"""Convex fixed-fibre tests for projection-generated action images.

Primal/dual values are floating-point diagnostics of exact-arithmetic SOCP
certificates. No exhaustive parameter-cell partition is used.
"""
import numpy as np
import cvxpy as cp


def fiber_certificate(latent, target_rows, anchor, radius=1., budget=0., sources=(),
                      target_equalities=None, active_tolerance=1e-9):
    """Distance from L V to anchor+N_K(anchor), with primal and dual solves.

    K={y:H y>=0,E y=0}; each source is (H_i,L_i), with K_i={x:H_i x>=0}.
    V={||v||<=radius, sum dist(L_i v,K_i polar)^2<=2*N*budget}.
    An empty source list represents a fully unobserved parameter ball.
    """
    L=np.asarray(latent,float);H=np.asarray(target_rows,float)
    a=np.asarray(anchor,float).ravel();q,p=L.shape
    H=H.reshape((-1,q));E=(np.empty((0,q)) if target_equalities is None
                            else np.asarray(target_equalities,float).reshape((-1,q)))
    if len(a)!=q or radius<=0 or budget<0:raise ValueError('Invalid dimensions/radius/budget')
    if len(H) and np.min(H@a)<-active_tolerance:raise ValueError('Anchor is outside the target cone')
    if len(E) and np.max(np.abs(E@a))>active_tolerance:raise ValueError('Target equality violated')
    slack=H@a;active=slack<=active_tolerance;A=H[active]
    norms=np.linalg.norm(H,axis=1);inactive=(~active)&(norms>0)
    margin=float(np.min(slack[inactive]/norms[inactive])) if inactive.any() else np.inf
    v=cp.Variable(p);constraints=[cp.norm(v)<=radius];normal=cp.Constant(np.zeros(q))
    if len(A):
        multipliers=cp.Variable(len(A),nonneg=True);normal=normal-A.T@multipliers
    if len(E):normal=normal+E.T@cp.Variable(len(E))
    errors=[];source_data=[]
    for rows,map_ in sources:
        rows=np.asarray(rows,float);map_=np.asarray(map_,float)
        rows=rows.reshape((-1,map_.shape[0]));source_data.append((rows,map_))
        error=map_@v
        if len(rows):error=error+rows.T@cp.Variable(len(rows),nonneg=True)
        errors.append(error)
    if errors:
        residual=cp.hstack(errors)
        if budget==0:constraints.append(residual==0)
        else:constraints.append(cp.norm(residual)<=np.sqrt(2*len(errors)*budget))
    primal=cp.Problem(cp.Minimize(cp.norm(L@v-a-normal)),constraints)
    primal.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=400)
    if primal.status not in ['optimal','optimal_inaccurate']:raise RuntimeError(primal.status)
    w=cp.Variable(q);dual_constraints=[cp.norm(w)<=1.]
    if len(A):dual_constraints.append(A@w<=0)
    if len(E):dual_constraints.append(E@w==0)
    b=L.T@w;variables=[]
    for rows,map_ in source_data:
        variable=cp.Variable(map_.shape[0]);variables.append(variable)
        if len(rows):dual_constraints.append(rows@variable>=0)
        b=b-map_.T@variable
    objective=a@w-radius*cp.norm(b)
    if variables and budget>0:
        objective=objective-np.sqrt(2*len(variables)*budget)*cp.norm(cp.hstack(variables))
    dual=cp.Problem(cp.Maximize(objective),dual_constraints)
    dual.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=400)
    if dual.status not in ['optimal','optimal_inaccurate']:raise RuntimeError(dual.status)
    W=np.asarray(w.value).ravel();W/=max(1.,np.linalg.norm(W))
    Q=[np.asarray(variable.value).ravel()/max(1.,np.linalg.norm(w.value)) for variable in variables]
    b=L.T@W
    for (_,map_),value in zip(source_data,Q):b-=map_.T@value
    raw=float(a@W-radius*np.linalg.norm(b))
    if Q and budget>0:raw-=np.sqrt(2*len(Q)*budget)*np.linalg.norm(np.concatenate(Q))
    violation=max([0.,float(np.max(A@W)) if len(A) else 0.,
                   float(np.max(np.abs(E@W))) if len(E) else 0.]+
                  [float(max(0.,-np.min(rows@value))) for (rows,_),value in zip(source_data,Q) if len(rows)])
    if violation>1e-7:raise RuntimeError('Dual feasibility residual exceeds the numerical acceptance tolerance')
    guard=1e-7*max(1.,np.linalg.norm(a),abs(raw))
    lower=max(0.,raw-guard);upper=float(primal.value)
    # The exact-arithmetic lower formula requires feasible w and q. Residuals
    # and the backoff are recorded; they do not constitute interval validation.
    return dict(lower=lower,upper=upper,margin=margin,
                distance_lower=min(margin,lower),distance_upper=upper,
                local_identity_region=bool(upper<margin),
                parameter=np.asarray(v.value).ravel(),dual_w=W,dual_q=Q,
                primal_status=primal.status,dual_status=dual.status,
                dual_feasibility_residual=violation,raw_dual=raw,numerical_backoff=guard)


def finite_witness_gap(certificate, variance_lower, mixed_upper):
    """Use an anchor equal to the mean of the feasible contact witnesses."""
    lower=float(variance_lower+certificate['distance_lower']**2-mixed_upper)
    return dict(single_minus_mixed_lower=lower,strict_gain=bool(lower>0))


def global_fiber_primal(latent,offset,target_rows,target_bounds,anchor,prior_center,
                        radius,sources,budget=0.):
    """SOCP primal for affine targets and bounded/nonconic source polyhedra.

    Target H y>=h. Each source dictionary specifies A x<=b, a feasible label y,
    and affine latent c+L v, under keys A,b,y,L,c. Returns an upper distance
    value, not the homogeneous explicit dual certificate supplied above.
    """
    L=np.asarray(latent,float);c=np.asarray(offset,float)
    H=np.asarray(target_rows,float).reshape((-1,L.shape[0]))
    h=np.asarray(target_bounds,float);a=np.asarray(anchor,float)
    if radius<=0 or budget<0 or (len(H) and np.min(H@a-h)<-1e-9):raise ValueError('Invalid prior/budget or infeasible target anchor')
    active=np.abs(H@a-h)<=1e-9;v=cp.Variable(L.shape[1])
    constraints=[cp.norm(v-np.asarray(prior_center,float))<=radius]
    normal=cp.Constant(np.zeros(L.shape[0]))
    if active.any():normal=-H[active].T@cp.Variable(int(active.sum()),nonneg=True)
    losses=[]
    for item in sources:
        A=np.asarray(item['A'],float);b=np.asarray(item['b'],float);y=np.asarray(item['y'],float)
        latent_i=np.asarray(item['L'],float)@v+np.asarray(item['c'],float)
        slack=b-A@y
        if np.min(slack)<-1e-9:raise ValueError('Source label is infeasible')
        if budget==0:
            rows=A[slack<=1e-9]
            residual=latent_i-y
            if len(rows):residual=residual-rows.T@cp.Variable(len(rows),nonneg=True)
            constraints.append(residual==0)
        else:
            multipliers=cp.Variable(len(A),nonneg=True)
            losses.append(.5*cp.sum_squares(latent_i-y-A.T@multipliers)+slack@multipliers)
    if losses:constraints.append(cp.sum(cp.hstack(losses))<=len(sources)*budget)
    problem=cp.Problem(cp.Minimize(cp.norm(c+L@v-a-normal)),constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=400)
    if problem.status not in ['optimal','optimal_inaccurate']:raise RuntimeError(problem.status)
    return dict(upper=float(problem.value),parameter=np.asarray(v.value),status=problem.status)
