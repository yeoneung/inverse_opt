"""Planar separation and optimal recovery for polyhedral projection responses.

The finite-candidate oracle is exact in exact arithmetic. Polynomial roots,
quadratic programs, and the enclosing-ball solve here use floating point;
reported intervals are numerical diagnostics, not interval-arithmetic proofs.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


@dataclass
class LinearRegion:
    rows: np.ndarray                 # rows @ parameter >= 0
    action: np.ndarray               # action @ parameter


@dataclass
class Response:
    regions: list

    def matrix_at(self, parameter):
        v=np.asarray(parameter,float)
        for region in self.regions:
            if np.all(region.rows@v >= -1e-10*max(1.,np.linalg.norm(v))):
                return region.action
        raise ValueError('Response regions do not cover the parameter')

    def __call__(self, parameter):
        return self.matrix_at(parameter)@parameter


@dataclass
class QuadraticCell:
    rows: np.ndarray
    information: np.ndarray
    action: np.ndarray
    lower_angle: float = 0.
    upper_angle: float = 2*np.pi


def linear_response(matrix):
    matrix=np.atleast_2d(np.asarray(matrix,float))
    if matrix.shape[1]!=2:raise ValueError('The implemented oracle has two parameters')
    return Response([LinearRegion(np.empty((0,2)),matrix)])


def negative_halfline_response(row):
    row=np.asarray(row,float).reshape(1,2)
    return Response([LinearRegion(-row,row),LinearRegion(row,np.zeros_like(row))])


def cone_projection_response(normals, latent_map=None):
    """Euclidean projection onto {u: normals @ u <= 0} in two action dimensions."""
    H=np.asarray(normals,float);L=np.eye(2) if latent_map is None else np.asarray(latent_map,float)
    if H.shape!=(2,2) or L.shape!=(2,2) or abs(np.linalg.det(H))<1e-9:
        raise ValueError('Two independent action normals and a planar latent map are required')
    regions=[LinearRegion(-H@L,L)]
    for j in range(2):
        normal=H[j]/np.linalg.norm(H[j]);ray=np.array([-normal[1],normal[0]])
        if H[1-j]@ray>0:ray=-ray
        P=np.outer(ray,ray)
        regions.append(LinearRegion(np.vstack((ray@L,normal@L)),P@L))
    T=np.linalg.solve(H@H.T,H)
    regions.append(LinearRegion(T@L,np.zeros((2,2))))
    return Response(regions)


def common_cells(source, target):
    """Build a common angular refinement without a Cartesian product of faces."""
    if not source or not target:raise ValueError('Source and target contexts are required')
    angles=[0.,np.pi/2,np.pi,3*np.pi/2]
    for response in list(source)+list(target):
        for region in response.regions:
            for row in region.rows:
                if np.linalg.norm(row)>1e-12:
                    angle=float(np.arctan2(row[0],-row[1])%(2*np.pi))
                    angles.extend((angle,(angle+np.pi)%(2*np.pi)))
    angles=sorted(angles);unique=[]
    for a in angles:
        if not unique or a-unique[-1]>1e-10:unique.append(a)
    if 2*np.pi-unique[-1]<1e-10:unique.pop()
    result=[]
    for i,a in enumerate(unique):
        b=unique[(i+1)%len(unique)]+(2*np.pi if i==len(unique)-1 else 0.)
        mid=np.array([np.cos((a+b)/2),np.sin((a+b)/2)])
        rows=np.array([[-np.sin(a),np.cos(a)],[np.sin(b),-np.cos(b)]])
        src=[s.matrix_at(mid) for s in source]
        I=sum(A.T@A for A in src)/len(source)
        A=np.vstack([t.matrix_at(mid) for t in target])/np.sqrt(len(target))
        result.append(QuadraticCell(rows,(I+I.T)/2,A,a,b))
    return result


def _unit(angle):return np.array([np.cos(angle),np.sin(angle)])


def _boundary_angles(rows):
    result=[0.,np.pi/2,np.pi,3*np.pi/2]
    for row in rows:
        if np.linalg.norm(row)>1e-13:
            angle=float(np.arctan2(row[0],-row[1])%(2*np.pi))
            result.extend((angle,(angle+np.pi)%(2*np.pi)))
    ordered=sorted(set(result))
    result+= [(a+(ordered[(i+1)%len(ordered)]+(2*np.pi if i==len(ordered)-1 else 0.)))/2
              for i,a in enumerate(ordered)]
    return result


def _real_roots(coefficients):
    coefficients=np.asarray(coefficients,float)
    scale=float(np.max(np.abs(coefficients)))
    if scale==0:return []
    coefficients=coefficients/scale
    while len(coefficients)>1 and abs(coefficients[0])<1e-12:coefficients=coefficients[1:]
    if len(coefficients)<=1:return []
    return [float(z.real) for z in np.roots(coefficients) if abs(z.imag)<1e-7*(1+abs(z.real))]


def _stationary_angles(H,b):
    """Stationarity of v(theta)' H v(theta) - 2 b' v(theta)."""
    d=H[1,1]-H[0,0];h=H[0,1]
    polynomial=[h+b[1],-2*d+2*b[0],-6*h,2*d+2*b[0],h-b[1]]
    return [0.,np.pi]+[float((2*np.arctan(t))%(2*np.pi)) for t in _real_roots(polynomial)]


def _quadratic_level_angles(I,level):
    mean=(I[0,0]+I[1,1])/2;cosine=(I[0,0]-I[1,1])/2;sine=I[0,1]
    amplitude=float(np.hypot(cosine,sine))
    if amplitude<1e-13*max(1.,np.linalg.norm(I)):return []
    ratio=(level-mean)/amplitude
    if abs(ratio)>1+1e-10:return []
    phase=np.arctan2(sine,cosine);offset=np.arccos(np.clip(ratio,-1,1))
    return [float(((phase+sign*offset)/2+k*np.pi)%(2*np.pi)) for sign in [-1,1] for k in [0,1]]


def farthest_in_cell(cell,center,radius,budget):
    """All candidate extrema on rays, circle, and training ellipse; p=2."""
    if radius<=0 or budget<0:raise ValueError('Positive radius and nonnegative loss budget required')
    a=np.asarray(center,float);I=cell.information;A=cell.action;c=2*budget
    eigen,U=np.linalg.eigh(I);scale=max(1.,float(eigen[-1]));rank=int(np.sum(eigen>1e-11*scale))
    points=[np.zeros(2)]
    def add(x):
        x=np.asarray(x,float)
        if (np.all(cell.rows@x>=-2e-9*max(1.,np.linalg.norm(x))) and
            x@x<=radius**2+2e-9*max(1.,radius**2) and
            x@I@x<=c+2e-9*max(1.,c)):
            points.append(x)
    if c==0 and rank==2:
        pass
    elif c==0 and rank==1:
        null=U[:,0];add(radius*null);add(-radius*null)
    else:
        for theta in _boundary_angles(cell.rows):
            v=_unit(theta);energy=max(0.,float(v@I@v))
            maximum=radius if energy<=1e-14 else min(radius,np.sqrt(c/energy))
            add(maximum*v)
        H=radius**2*A.T@A;b=radius*A.T@a
        for theta in _stationary_angles(H,b)+_quadratic_level_angles(I,c/radius**2):add(radius*_unit(theta))
        if c>0 and rank==2:
            E=(U*np.sqrt(c/eigen))@U.T;AE=A@E
            for theta in _stationary_angles(AE.T@AE,AE.T@a):add(E@_unit(theta))
    values=[float(np.sum((A@x-a)**2)) for x in points];j=int(np.argmax(values))
    return dict(value=values[j],parameter=points[j],action=A@points[j],candidates=len(points))


def farthest_action(cells,center,radius,budget):
    choices=[farthest_in_cell(cell,center,radius,budget) for cell in cells]
    j=int(np.argmax([c['value'] for c in choices]));result=dict(choices[j]);result['cell']=j
    result['candidates']=sum(c['candidates'] for c in choices)
    return result


def minimum_enclosing_ball(points):
    """Feasible simplex dual gives a lower bound even with approximate optimization."""
    Y=np.asarray(points,float);n=len(Y);squared=np.sum(Y**2,axis=1)
    def objective(w):return float(np.sum((w@Y)**2)-w@squared)
    def gradient(w):return 2*Y@(w@Y)-squared
    solve=minimize(objective,np.full(n,1/n),jac=gradient,method='SLSQP',bounds=[(0,1)]*n,
        constraints={'type':'eq','fun':lambda w:float(w.sum()-1),'jac':lambda w:np.ones(n)},
        options={'ftol':1e-13,'maxiter':2000})
    weights=np.maximum(solve.x,0.);weights/=weights.sum();center=weights@Y
    lower=max(0.,float(weights@squared-center@center))
    old_upper=float(np.max(np.sum((Y-center)**2,axis=1)))
    # Refine the active support's affine KKT system. The feasible dual weights
    # remain a valid lower witness whether or not this refinement is accepted.
    support=np.flatnonzero(weights>1e-10);seen=set()
    for _ in range(2*n+2):
        signature=tuple(support)
        if not len(support) or signature in seen:break
        seen.add(signature)
        Z=Y[support];k=len(support)
        system=np.block([[2*Z@Z.T,np.ones((k,1))],[np.ones((1,k)),np.zeros((1,1))]])
        rhs=np.r_[squared[support],1.]
        refined=np.linalg.lstsq(system,rhs,rcond=None)[0][:k]
        if refined.min()<-1e-10:
            support=np.delete(support,int(np.argmin(refined)));continue
        refined=np.maximum(refined,0.);refined/=refined.sum();trial=np.zeros(n);trial[support]=refined
        c=trial@Y;lo=max(0.,float(trial@squared-c@c));distances=np.sum((Y-c)**2,axis=1);hi=float(distances.max())
        if hi-lo<old_upper-lower:
            weights=trial;center=c;lower=lo;old_upper=hi
        if hi-lo<=1e-12*max(1.,hi):break
        violated=int(np.argmax(distances))
        if violated in support:break
        support=np.r_[support,violated]
    return dict(center=center,lower=lower,weights=weights,restricted_upper=old_upper,
                success=bool(solve.success),restricted_gap=max(0.,old_upper-lower))


def optimal_recovery(cells,radius,budget,absolute_tolerance=1e-7,max_iterations=300):
    if not cells or absolute_tolerance<=0 or max_iterations<1:raise ValueError('Cells, positive tolerance and at least one iteration required')
    q=cells[0].action.shape[0];points=[np.zeros(q)];parameters=[np.zeros(2)];history=[]
    for iteration in range(max_iterations):
        ball=minimum_enclosing_ball(points);oracle=farthest_action(cells,ball['center'],radius,budget)
        lower=ball['lower'];upper=max(lower,oracle['value'])
        history.append(dict(iteration=iteration,lower=lower,upper=upper,
                            restricted_gap=ball['restricted_gap'],oracle_candidates=oracle['candidates']))
        if upper-lower<=absolute_tolerance:
            return dict(center=ball['center'],lower=lower,upper=upper,converged=True,history=history,
                        points=np.array(points),parameters=np.array(parameters),weights=ball['weights'])
        if iteration+1==max_iterations or min(np.linalg.norm(oracle['action']-p) for p in points)<=1e-10:
            break
        points.append(oracle['action']);parameters.append(oracle['parameter'])
    return dict(center=ball['center'],lower=lower,upper=upper,converged=False,history=history,
                points=np.array(points),parameters=np.array(parameters),weights=ball['weights'])


def sharp_transfer_constant_2d(cells):
    best=0.;direction=np.zeros(2)
    for cell in cells:
        I=cell.information;B=cell.action.T@cell.action
        angles=_boundary_angles(cell.rows)
        polynomial=[B[1,1]*I[0,1]-B[0,1]*I[1,1],
                    B[1,1]*I[0,0]-B[0,0]*I[1,1],
                    B[0,1]*I[0,0]-B[0,0]*I[0,1]]
        for slope in _real_roots(polynomial):
            theta=np.arctan(slope);angles.extend((theta,theta+np.pi))
        eigen,U=np.linalg.eigh(I)
        for j,value in enumerate(eigen):
            if value<1e-11*max(1.,eigen[-1]):
                theta=np.arctan2(U[1,j],U[0,j]);angles.extend((theta,theta+np.pi))
        for theta in angles:
            v=_unit(theta)
            if np.any(cell.rows@v<-1e-8):continue
            den=max(0.,float(v@I@v));num=max(0.,float(v@B@v))
            if den<=1e-11*max(1.,np.linalg.norm(I)):
                if num>1e-9*max(1.,np.linalg.norm(B)):return dict(value=float('inf'),direction=v)
                continue
            ratio=num/den
            if ratio>best:best=ratio;direction=v
    return dict(value=best,direction=direction)


def cone_sector_recovery(beta,axis,radius=1.):
    """Closed form for an isometric target on a cone-ball uncertainty set."""
    axis=np.asarray(axis,float)
    if radius<=0 or not 0<=beta<=1:raise ValueError('Invalid radius or cone angular margin')
    if beta==0:return dict(center=np.zeros_like(axis),risk=radius**2)
    if not np.isclose(np.linalg.norm(axis),1.):raise ValueError('Unit axis required')
    if beta>=1/np.sqrt(2):distance=radius/(2*beta);risk=distance**2
    else:distance=radius*beta;risk=radius**2*(1-beta**2)
    return dict(center=distance*axis,risk=float(risk))


def scalar_cone_recovery(rows,target,radius=1.):
    """Two convex cone projections for a scalar target, in any dimension."""
    D=np.asarray(rows,float);b=np.asarray(target,float).ravel();p=len(b)
    if D.ndim!=2 or D.shape[1]!=p or radius<=0:raise ValueError('Invalid cone or radius')
    def projection(z):
        if len(D)==0:return z.copy()
        solve=minimize(lambda x:.5*np.sum((x-z)**2),np.zeros(p),jac=lambda x:x-z,
            constraints={'type':'ineq','fun':lambda x:D@x,'jac':lambda x:D},
            method='SLSQP',options={'ftol':1e-13,'maxiter':1000})
        if not solve.success:raise RuntimeError(solve.message)
        return solve.x
    positive=projection(b);negative=projection(-b)
    hi=radius*np.linalg.norm(positive);lo=-radius*np.linalg.norm(negative)
    return dict(center=float((hi+lo)/2),risk=float(((hi-lo)/2)**2),
                interval=[float(lo),float(hi)],positive_projection=positive,negative_projection=negative)


def isometric_cone_from_rays(generators,radius=1.):
    """Reduce cone-ball recovery to one finite enclosing ball, for an isometry."""
    rays=np.asarray(generators,float)
    if rays.ndim!=2 or radius<=0:raise ValueError('Invalid rays or radius')
    lengths=np.linalg.norm(rays,axis=1);rays=rays[lengths>1e-14]
    if not len(rays):return dict(center=np.zeros(rays.shape[1]),risk=0.,upper=0.)
    rays=rays/np.linalg.norm(rays,axis=1)[:,None]
    ball=minimum_enclosing_ball(np.vstack((np.zeros((1,rays.shape[1])),radius*rays)))
    return dict(center=ball['center'],risk=ball['lower'],upper=ball['restricted_upper'],weights=ball['weights'])
