"""Fixed, genuinely multi-directional checks of convex fixed-fibre certificates."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.fiber_certificate import fiber_certificate,finite_witness_gap
from src.optimal_recovery import minimum_enclosing_ball


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='fiber')
    args=parser.parse_args()
    if Path(args.output).name!=args.output:raise ValueError('Use a simple output name')
    out=ROOT/'results'/args.output
    if out.exists():raise FileExistsError('Preserve previous results')
    out.mkdir(parents=True)
    protocol=ROOT/'configs/fiber_protocol.json';cfg=json.loads(protocol.read_text())
    rows=[];details={};r=cfg['radius']
    for d in cfg['invisible_dimensions']:
        for seed in cfg['seeds']:
            rng=np.random.default_rng(seed+100*d)
            S=np.linalg.qr(rng.normal(size=(d+1,d+1)))[0];Z=S[:,:d];n=S[:,d]
            R=np.linalg.qr(rng.normal(size=(d,d)))[0]
            sources=[(np.array([[-1.]]),sign*n.reshape(1,-1)) for sign in [1.,-1.]]
            for kind in cfg['targets']:
                q=2*d if kind=='split' else d
                U=np.linalg.qr(rng.normal(size=(q,q)))[0]
                base=np.vstack((np.eye(d),-np.eye(d))) if kind=='split' else np.eye(d)
                L=U@base@R@Z.T;H=U.T
                anchor=U@(np.ones(q)*r/q)
                points=r*U.T
                expected_R=r*r*(1-1/q);expected_P=r*r if kind=='split' else expected_R
                expected_delta=r/np.sqrt(q) if kind=='split' else 0.
                ball=minimum_enclosing_ball(np.vstack((np.zeros(q),points)))
                parameters=rng.normal(size=(cfg['sampled_hidden_parameters'],d))
                parameters/=np.linalg.norm(parameters,axis=1)[:,None]
                parameters*=r*rng.uniform(.05,1.,size=(len(parameters),1))
                sample=parameters@Z.T
                latent=sample@L.T;actions=np.maximum(latent@U,0)@U.T
                sample_error=float(np.max(np.sum((actions-anchor)**2,axis=1)))
                nnls_error=0.
                for z,y in zip(latent[:cfg['independent_nnls_parameters']],actions):
                    coefficients,_=nnls(U,z)
                    nnls_error=max(nnls_error,float(np.linalg.norm(U@coefficients-y)))
                for budget in cfg['source_loss_budgets']:
                    start=time.perf_counter()
                    cert=fiber_certificate(L,H,anchor,r,budget,sources)
                    seconds=time.perf_counter()-start
                    gap=finite_witness_gap(cert,expected_R,expected_R)
                    name=f'{d}_{seed}_{kind}_{budget:g}'
                    rows.append(dict(name=name,invisible_dimension=d,parameter_dimension=d+1,
                        action_dimension=q,seed=seed,target=kind,budget=budget,
                        delta_lower=cert['lower'],delta_upper=cert['upper'],delta_exact=expected_delta,
                        delta_error=abs(cert['upper']-expected_delta),dual_gap=cert['upper']-cert['lower'],
                        dual_feasibility_residual=cert['dual_feasibility_residual'],
                        margin=cert['margin'],mixed_risk=expected_R,single_risk=expected_P,
                        squared_gap_lower=gap['single_minus_mixed_lower'],strict_gain=gap['strict_gain'],
                        exact_squared_gap=expected_P-expected_R,
                        ball_formula_error=abs(ball['restricted_upper']-expected_R),
                        center_formula_error=float(np.linalg.norm(ball['center']-anchor)),
                        sampled_excess=max(0.,sample_error-expected_R),nnls_error=nnls_error,seconds=seconds))
                    details[name]=dict(latent=L.tolist(),target_rows=H.tolist(),source_normal=n.tolist(),
                        anchor=anchor.tolist(),primal_parameter=cert['parameter'].tolist(),
                        dual_w=cert['dual_w'].tolist(),dual_q=[v.tolist() for v in cert['dual_q']],
                        primal_status=cert['primal_status'],dual_status=cert['dual_status'])
            print('dimension',d,'seed',seed,'complete',flush=True)
    with (out/'runs.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    metadata=dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        cases=len(rows),details=details,interpretation=cfg['interpretation'])
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    analysis=dict(cases=len(rows),split_cases=sum(row['target']=='split' for row in rows),
        positive_split_certificates=sum(row['target']=='split' and row['strict_gain'] for row in rows),
        false_positive_controls=sum(row['target']=='orthant' and row['strict_gain'] for row in rows),
        maximum_delta_error=max(row['delta_error'] for row in rows),
        maximum_dual_gap=max(row['dual_gap'] for row in rows),
        maximum_dual_feasibility_residual=max(row['dual_feasibility_residual'] for row in rows),
        maximum_ball_formula_error=max(row['ball_formula_error'] for row in rows),
        maximum_center_formula_error=max(row['center_formula_error'] for row in rows),
        maximum_nnls_error=max(row['nnls_error'] for row in rows),
        maximum_sampled_excess=max(row['sampled_excess'] for row in rows))
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':main()
