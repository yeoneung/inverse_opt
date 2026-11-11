"""Global line-projection paths and finite-image recovery.

Polyhedral response intervals are inferred from forward QPs and checked by
their affine KKT inequalities. Numerical results are double-precision
diagnostics; exact_chain_check uses rational arithmetic for the lift identity.
"""
from fractions import Fraction as Q
import numpy as np
import cvxpy as cp

def dot(a,b):return sum(x*y for x,y in zip(a,b))

def sub(a,b):return tuple(x-y for x,y in zip(a,b))

def add(a,b):return tuple(x+y for x,y in zip(a,b))

def mul(t,a):return tuple(t*x for x in a)

def exact_chain_check(points,delta):
    points=[tuple(Q(x) for x in p) for p in points]
    n=len(points)-1
    vertices=[p+(delta*i,delta*i*i) for i,p in enumerate(points)]
    bound=max(dot(v,v) for v in vertices)
    M=2*(n+1)*bound+1
    intervals=[];maximum=Q(0);smallest_gap=None
    for i in range(n):
        edge=sub(vertices[i+1],vertices[i])
        interval=tuple(Q(2*i+1)+dot(v,edge)/M for v in vertices[i:i+2])
        assert interval[0]<interval[1]
        if intervals:
            gap=interval[0]-intervals[-1][1]
            assert gap>0
            smallest_gap=gap if smallest_gap is None else min(smallest_gap,gap)
        intervals.append(interval)
        for s in [Q(0),Q(1,3),Q(1,2),Q(1)]:
            x=add(vertices[i],mul(s,edge))
            lam=Q(2*i+1)+dot(x,edge)/M
            latent=(Q(0),)*len(points[0])+(M*lam/delta,-M/delta)
            normal=sub(latent,x)
            assert dot(normal,edge)==0
            for j,v in enumerate(vertices):
                residual=dot(normal,sub(v,x))
                if j in [i,i+1]:assert residual==0
                else:assert residual<0
                maximum=max(maximum,residual)
    return dict(vertices=len(vertices),dimension=len(vertices[0]),maximum_normal_violation=str(maximum),
                minimum_vertex_plateau=str(smallest_gap) if smallest_gap is not None else None)

def float_lift(points,delta):
    n=len(points)-1;index=np.arange(n+1,dtype=float)
    vertices=np.column_stack((points,delta*index,delta*index*index))
    # Compute the exact finite sufficient threshold for this floating-point case.
    threshold=0.
    for i in range(n):
        edge=vertices[i+1]-vertices[i]
        for j in range(n+1):
            k=j-i
            if k in [0,1]:continue
            remainder=k*edge-(vertices[j]-vertices[i])
            threshold=max(threshold,max(vertices[i]@remainder,vertices[i+1]@remainder)/(k*(k-1)))
    M=max(1.,2*threshold+1.)
    intervals=[]
    for i in range(n):
        edge=vertices[i+1]-vertices[i]
        intervals.append((2*i+1+vertices[i]@edge/M,2*i+1+vertices[i+1]@edge/M))
    return vertices,M,np.array(intervals)

def proper_radius(vertices):
    vertices=np.asarray(vertices,dtype=float)
    if vertices.ndim!=2 or not len(vertices):raise ValueError('Nonempty vertex array required')
    best=float(np.max(np.sum((vertices-vertices[0])**2,axis=1)));center=vertices[0].copy()
    for start,end in zip(vertices[:-1],vertices[1:]):
        edge=end-start;a=edge@edge
        if a<=1e-28:continue
        b=2*(start-vertices)@edge;c=np.sum((start-vertices)**2,axis=1)
        pair=np.triu_indices(len(vertices),1);den=b[pair[0]]-b[pair[1]]
        valid=np.abs(den)>1e-14
        crossings=(c[pair[1]][valid]-c[pair[0]][valid])/den[valid]
        candidates=np.r_[0.,1.,-b/(2*a),crossings]
        candidates=candidates[(candidates>=0)&(candidates<=1)]
        values=a*candidates*candidates+np.max(b[:,None]*candidates+c[:,None],axis=0)
        j=np.argmin(values)
        if values[j]<best:best=float(values[j]);center=start+candidates[j]*edge
    return best,center

def mixed_radius(vertices):
    vertices=np.asarray(vertices,dtype=float)
    if vertices.ndim!=2 or not len(vertices):raise ValueError('Nonempty vertex array required')
    if len(vertices)==1:return 0.,0.,vertices[0].copy()
    center=cp.Variable(vertices.shape[1]);t=cp.Variable()
    constraints=[cp.sum_squares(center-v)<=t for v in vertices]
    problem=cp.Problem(cp.Minimize(t),constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
    weights=np.array([float(c.dual_value[0]) for c in constraints]);weights=np.maximum(weights,0);weights/=weights.sum()
    mean=weights@vertices
    lower=float(weights@np.sum((vertices-mean)**2,axis=1))
    # Use the contact mean as the supplied anchor. This guarantees membership
    # in conv(vertices) and makes the finite-witness mean condition exact.
    upper=float(np.max(np.sum((vertices-mean)**2,axis=1)))
    return lower,upper,mean

def qp_check(vertices,M,intervals,edge_indices):
    index=np.arange(len(vertices),dtype=float);weights=cp.Variable(len(vertices),nonneg=True)
    costs=cp.Parameter(len(vertices));point=vertices.T@weights
    problem=cp.Problem(cp.Minimize(.5*cp.sum_squares(point)+costs@weights),[cp.sum(weights)==1])
    errors=[];normal_violations=[]
    for i in edge_indices:
        for fraction in [.17,.5,.83]:
            lam=(1-fraction)*intervals[i,0]+fraction*intervals[i,1]
            # The omitted scalar is independent of weights. This objective is
            # exactly the full-polytope projection QP, with better scaling.
            raw=M*(index-lam/2)**2
            costs.value=raw-raw.min()
            problem.solve(solver='CLARABEL',tol_gap_abs=1e-12,tol_gap_rel=1e-12,tol_feas=1e-12,max_iter=400)
            if problem.status not in ['optimal','optimal_inaccurate']:raise RuntimeError(problem.status)
            expected=(1-fraction)*vertices[i]+fraction*vertices[i+1]
            errors.append(float(np.linalg.norm(point.value-expected)))
            k=index-i;edge=vertices[i+1]-vertices[i]
            remainder=k[:,None]*edge-(vertices-vertices[i])
            normal=-M*k*(k-1)+remainder@expected
            normal_violations.append(float(normal.max()))
    return max(errors),max(normal_violations)

def polyhedral_line_path(A,b,c,d,left,right,return_pieces=False):
    t=cp.Parameter();x=cp.Variable(len(c))
    qp=cp.Problem(cp.Minimize(.5*cp.sum_squares(x-c-t*d)),[A@x<=b])
    pending=[(left,right)];pieces=[];iterations=0;max_residual=0.
    while pending:
        lo,hi=pending.pop()
        if hi-lo<=1e-9:continue
        iterations+=1
        if iterations>200:raise RuntimeError('Path partition iteration limit')
        mid=(lo+hi)/2;t.value=mid
        qp.solve(solver='CLARABEL',tol_gap_abs=1e-11,tol_gap_rel=1e-11,tol_feas=1e-11,max_iter=400)
        if qp.status not in ['optimal','optimal_inaccurate']:raise RuntimeError(qp.status)
        slack=b-A@x.value
        # Generate candidate faces from approximate primal/dual supports, then
        # accept solely through their full affine KKT feasibility interval.
        # This avoids treating a single numerical slack cutoff as an oracle.
        candidates=[]
        for cutoff in [2e-7,1e-4,1e-6,1e-8]:
            active=np.flatnonzero((slack<cutoff)&(np.asarray(qp.constraints[0].dual_value)>1e-7))
            if tuple(active) in candidates:continue
            candidates.append(tuple(active));rows=A[active]
            if len(active):
                gram=rows@rows.T
                if np.linalg.matrix_rank(gram,tol=1e-10)<len(active):continue
                multiplier0=np.linalg.solve(gram,rows@c-b[active])
                multiplier1=np.linalg.solve(gram,rows@d)
                intercept=c-rows.T@multiplier0;slope=d-rows.T@multiplier1
            else:
                multiplier0=np.empty(0);multiplier1=np.empty(0);intercept=c.copy();slope=d.copy()
            constant=np.r_[A@intercept-b,-multiplier0]
            linear=np.r_[A@slope,-multiplier1]
            lower,upper=left,right;invalid=False
            for a,s in zip(constant,linear):
                if s>1e-11:upper=min(upper,-a/s)
                elif s<-1e-11:lower=max(lower,-a/s)
                elif a>2e-7:invalid=True;break
            lower=max(left,lower);upper=min(right,upper)
            error=np.linalg.norm(intercept+mid*slope-x.value)
            if invalid or not lower-2e-7<=mid<=upper+2e-7 or upper-lower<1e-10 or error>2e-6:continue
            break
        else:raise RuntimeError('No candidate face passed the complete KKT interval check')
        for value in [lower,upper]:
            residual=np.r_[A@(intercept+value*slope)-b,-(multiplier0+value*multiplier1)]
            max_residual=max(max_residual,float(np.max(residual)))
        pieces.append((lower,upper,intercept,slope,active))
        if lower-lo>1e-8:pending.append((lo,min(lower,hi)))
        if hi-upper>1e-8:pending.append((max(upper,lo),hi))
    pieces.sort(key=lambda p:p[0]);unique=[]
    for piece in pieces:
        if unique and piece[0]<=unique[-1][1]+1e-8 and np.linalg.norm(piece[2]-unique[-1][2])+np.linalg.norm(piece[3]-unique[-1][3])<1e-6:
            previous=unique.pop();piece=(previous[0],max(previous[1],piece[1]),piece[2],piece[3],piece[4])
        unique.append(piece)
    assert abs(unique[0][0]-left)<1e-7 and abs(unique[-1][1]-right)<1e-7
    gap=max([0.]+[p[0]-q[1] for q,p in zip(unique[:-1],unique[1:])])
    assert gap<1e-7
    vertices=[unique[0][2]+unique[0][0]*unique[0][3]]
    for lo,hi,a,s,_ in unique:
        point=a+hi*s
        if np.linalg.norm(point-vertices[-1])>1e-9:vertices.append(point)
    result=(np.array(vertices),iterations,max_residual,gap)
    return result+(unique,) if return_pieces else result
