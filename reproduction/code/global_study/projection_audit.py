"""Independent audit of one weakly active forward projection; saved runs untouched."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))

import gurobipy as gp
import numpy as np
from scipy import sparse
from src.multidimensional_certification import Projector


def kkt(A,b,z,y,lam):
    slack=b-A@y
    return dict(primal=max(0.,float(np.max(-slack))),
                dual=max(0.,float(np.max(-lam))),
                stationarity=float(np.linalg.norm(y-z+A.T@lam,np.inf)),
                complementarity=float(np.max(np.abs(lam*slack))))


class GurobiAuditProjector:
    """Independent reusable primal-simplex convex-QP projection.

    Calling the object returns (response, nonnegative inequality duals).
    The returned diagnostics use the original unscaled KKT equations.
    This is a floating-point audit, not an interval certificate.
    """
    def __init__(self,A,b,acceptance_tolerance=1e-8):
        self.A=np.asarray(A,float)
        self.b=np.asarray(b,float)
        self.q=self.A.shape[1]
        self.acceptance_tolerance=float(acceptance_tolerance)
        self.env=gp.Env(empty=True)
        self.env.setParam('OutputFlag',0)
        self.env.start()
        self.model=gp.Model('independent_projection_audit',env=self.env)
        self.model.Params.OutputFlag=0
        self.model.Params.Threads=1
        self.model.Params.Method=0
        self.model.Params.FeasibilityTol=1e-9
        self.model.Params.OptimalityTol=1e-9
        self.model.Params.NumericFocus=3
        self.variables=[self.model.addVar(lb=-gp.GRB.INFINITY,name=f'y{j}') for j in range(self.q)]
        self.constraints=[self.model.addConstr(gp.quicksum(float(self.A[i,j])*self.variables[j] for j in range(self.q) if self.A[i,j])<=float(self.b[i])) for i in range(len(self.b))]
        self.model.setObjective(.5*gp.quicksum(a*a for a in self.variables),gp.GRB.MINIMIZE)
        self.model.update()
        self.calls=0
        self.maximum_kkt_residual=0.
        self.last_kkt=None

    def __call__(self,preferred):
        z=np.asarray(preferred,float)
        if z.shape!=(self.q,):
            raise ValueError('Preferred-action dimension mismatch.')
        for variable,value in zip(self.variables,z):
            variable.Obj=-float(value)
        self.model.optimize()
        if self.model.Status!=gp.GRB.OPTIMAL:
            raise RuntimeError('Independent convex QP did not solve to optimality: '+str(self.model.Status))
        y=np.array([a.X for a in self.variables])
        lam=-np.array([a.Pi for a in self.constraints])
        self.last_kkt=kkt(self.A,self.b,z,y,lam)
        residual=max(self.last_kkt.values())
        self.maximum_kkt_residual=max(self.maximum_kkt_residual,residual)
        self.calls+=1
        if not np.isfinite(residual) or residual>self.acceptance_tolerance:
            raise RuntimeError('Independent projection KKT residual exceeds strict acceptance: '+str(self.last_kkt))
        return y,lam

    def close(self):
        self.model.dispose()
        self.env.dispose()

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()


def main():
    import osqp
    if (ROOT/'projection_audit.json').exists():
        raise FileExistsError('Preserve the recorded numerical audit. Use the reusable projector class or a fresh copy for a new diagnostic.')
    folder=ROOT/'compact_capacity_extended'
    saved=json.loads((folder/'results.json').read_text())
    case=next(c for c in saved['cases'] if c['instance']=='q12_p8_s846102_c1.4')
    history=next(h for h in case['history'] if h['iteration']==5)
    instance=np.load(folder/case['instance_file'])
    A,b,c,G=[instance[key] for key in ['A','b','c','G']]
    raw_v=np.asarray(history['oracle']['parameter']);v=np.clip(raw_v,-1,1)
    z=c+G@v;stored=np.asarray(history['oracle']['response']);anchor=np.asarray(history['anchor'])
    q=len(c)
    rows=[]
    def append(name,y,lam=None,status=None):
        y=np.asarray(y)
        record=dict(method=name,status=status,response=y.tolist(),
                    objective=float(.5*np.sum((y-z)**2)),risk=float(np.sum((y-anchor)**2)),
                    difference_from_stored_inf=float(np.linalg.norm(y-stored,np.inf)))
        if lam is not None:record['kkt']=kkt(A,b,z,y,np.asarray(lam))
        rows.append(record)
    append('stored',stored,status=history['oracle']['status'])
    ordinary=Projector(A,b,q);y,lam=ordinary(z);append('fresh_existing_projector',y,lam)
    if np.array_equal(A[:q],-np.eye(q)) and np.array_equal(A[q:2*q],np.eye(q)) and np.array_equal(b[:q],np.zeros(q)):
        clip=np.clip(z,0,b[q:2*q])
        clip_lam=np.r_[np.maximum(-z,0),np.maximum(z-b[q:2*q],0),np.zeros(len(b)-2*q)]
        append('analytical_box_projection',clip,clip_lam,status='exact formula if primal feasible')
    solver=osqp.OSQP()
    solver.setup(P=sparse.eye(q,format='csc'),q=-z,A=sparse.csc_matrix(A),
                 l=np.full(len(b),-np.inf),u=b,eps_abs=1e-12,eps_rel=1e-12,
                 max_iter=200000,polishing=True,verbose=False)
    solution=solver.solve()
    append('osqp_polished_1e-12',solution.x,solution.y,status=solution.info.status)
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        for method in [0,1,2]:
            with gp.Model(env=env) as model:
                model.Params.OutputFlag=0;model.Params.Threads=1
                model.Params.FeasibilityTol=1e-9;model.Params.OptimalityTol=1e-9
                model.Params.BarConvTol=1e-12;model.Params.Method=method
                model.Params.NumericFocus=3
                variables=[model.addVar(lb=-gp.GRB.INFINITY,name=f'y{j}') for j in range(q)]
                constraints=[model.addConstr(gp.quicksum(float(A[i,j])*variables[j] for j in range(q) if A[i,j])<=float(b[i])) for i in range(len(b))]
                model.setObjective(.5*gp.quicksum((variables[j]-float(z[j]))*(variables[j]-float(z[j])) for j in range(q)),gp.GRB.MINIMIZE)
                model.optimize()
                y=np.array([a.X for a in variables]);lam=-np.array([a.Pi for a in constraints])
                append(f'gurobi_method_{method}',y,lam,status=int(model.Status))
    record={'instance':case['instance'],'history_iteration':5,
            'raw_prior_violation':float(np.max(np.maximum(np.abs(raw_v)-1,0))),
            'parameter_clipping_inf':float(np.linalg.norm(raw_v-v,np.inf)),
            'raw_parameter':raw_v.tolist(),'clipped_parameter':v.tolist(),'preferred_action':z.tolist(),
            'anchor':anchor.tolist(),'rows':rows,
            'scope':'Saved experiment and its detection have not been changed by this audit.'}
    (ROOT/'projection_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    for row in rows:
        print(json.dumps({k:v for k,v in row.items() if k!='response'}),flush=True)


if __name__=='__main__':
    main()
