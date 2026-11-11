"""Fixed checks of mixture classification, sharp price and finite-budget persistence."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy.optimize import minimize_scalar,nnls
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.optimal_recovery import (minimum_enclosing_ball,optimal_recovery,common_cells,
    cone_projection_response,negative_halfline_response,linear_response)
from src.mixture_recovery import two_branch_recovery,consistent_model_recovery,mixture_budget_bound
from experiments.recovery import direct_grid


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def rotation(angle):return np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])


def independent_single(a,b):
    vertices=np.vstack((np.zeros_like(a),a,b));results=[]
    for endpoint in [a,b]:
        f=lambda t:float(np.max(np.sum((vertices-t*endpoint)**2,axis=1)))
        solved=minimize_scalar(f,bounds=(0.,1.),method='bounded',options={'xatol':1e-13})
        results.append(min(f(0.),f(1.),solved.fun))
    return min(results)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='mixture')
    parser.add_argument('--coupled-source',default='run_recovery',help='Saved planar configuration directory below results.')
    args=parser.parse_args()
    if Path(args.output).name!=args.output:raise ValueError('Use a simple output name')
    if Path(args.coupled_source).name!=args.coupled_source:raise ValueError('Use a simple source directory name')
    out=ROOT/'results'/args.output;out.mkdir(parents=True,exist_ok=True)
    if any(out.glob('*.csv')):raise FileExistsError('Preserve previous results')
    protocol=ROOT/'configs/mixture_protocol.json';cfg=json.loads(protocol.read_text())
    phase=[];embedding=[];cases=[];details={};r=cfg['radius']
    for k in cfg['phase_length_ratios']:
        for degrees in cfg['phase_angles_degrees']:
            angle=np.deg2rad(degrees);a=np.array([1.,0.]);b=k*np.array([np.cos(angle),np.sin(angle)])
            result=two_branch_recovery(a,b);ball=minimum_enclosing_ball(np.vstack((np.zeros(2),a,b)))
            single=independent_single(a,b)
            phase.append(dict(length_ratio=k,angle_degrees=degrees,cosine=result['cosine'],
                mixed_formula=result['mixed_risk'],mixed_computed=ball['restricted_upper'],
                single_formula=result['single_risk'],single_computed=single,
                mixed_error=abs(ball['restricted_upper']-result['mixed_risk']),
                single_error=abs(single-result['single_risk']),ratio=result['ratio'],
                predicted_strict_gain=result['strict_gain'],computed_squared_gap=single-ball['restricted_upper']))
    write_csv(out/'phase.csv',phase);print('Phase cases:',len(phase),flush=True)
    for dimension in cfg['embedding_dimensions']:
        for seed in cfg['embedding_seeds']:
            rng=np.random.default_rng(seed+100*dimension);U=np.linalg.qr(rng.normal(size=(dimension,2)))[0]
            for row in phase:
                k=row['length_ratio'];rho=row['cosine']
                if k==0 or abs(rho)>=1-1e-10 or rho < -k-1e-12:continue
                a=U[:,0];b=k*(rho*U[:,0]+np.sqrt(1-rho*rho)*U[:,1]);G=np.column_stack((a,b))
                latent=G@np.linalg.solve(G.T@G,np.array([1.,-k*k]))
                pos=G@nnls(G,latent)[0];neg=G@nnls(G,-latent)[0]
                result=two_branch_recovery(pos,neg)
                embedding.append(dict(dimension=dimension,seed=seed,length_ratio=k,cosine=rho,
                    positive_projection_error=float(np.linalg.norm(pos-a)),negative_projection_error=float(np.linalg.norm(neg-b)),
                    mixed_error=abs(result['mixed_risk']-row['mixed_formula']),
                    single_error=abs(result['single_risk']-row['single_formula']),ratio=result['ratio']))
    write_csv(out/'embeddings.csv',embedding);print('Projection realizations:',len(embedding),flush=True)

    def run_case(name,group,source,target,budget,bound=None,extra_directions=None):
        cells=common_cells(source,target)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            start=time.perf_counter();mixed=optimal_recovery(cells,r,budget,1e-7,300)
            mixed_seconds=time.perf_counter()-start
            start=time.perf_counter()
            try:
                single=consistent_model_recovery(cells,r,budget,
                    cfg['absolute_squared_gap'],cfg['maximum_exchange_iterations'],mixed['points'])
            except Exception as error:
                if cases:write_csv(out/'cases.csv',cases)
                (out/'failure.json').write_text(json.dumps(dict(name=name,group=group,budget=budget,
                    error=repr(error),completed_cases=len(cases),details=details),indent=2),encoding='utf-8')
                raise
            single_seconds=time.perf_counter()-start
        parameters,images=direct_grid(source,target,r,budget,cfg['independent_angular_grid'])
        if extra_directions is not None:
            extra=np.asarray(extra_directions)*r
            extra_images=np.hstack([np.vstack([t(v) for v in extra]) for t in target])/np.sqrt(len(target))
            images=np.vstack((images,extra_images))
        sampled_mixed=float(np.max(np.sum((images-mixed['center'])**2,axis=1)))
        sampled_single=float(np.max(np.sum((images-single['center'])**2,axis=1)))
        v=single['parameter'];source_loss=sum(float(np.sum(s(v)**2)) for s in source)/(2*len(source))
        statuses=[h['restricted_status'] for cell in single['cell_results'] for h in cell['history']]
        row=dict(name=name,group=group,budget=budget,cells=len(cells),
            mixed_lower=mixed['lower'],mixed_upper=mixed['upper'],mixed_converged=mixed['converged'],
            single_lower=single['lower'],single_upper=single['upper'],single_converged=single['converged'],
            single_gap=single['upper']-single['lower'],price_lower=single['lower']/mixed['upper'] if mixed['upper'] else 1.,
            price_upper=single['upper']/mixed['lower'] if mixed['lower']>0 else np.inf,
            observed_squared_gap_lower=single['lower']-mixed['upper'],
            grid_excess=max(0.,sampled_mixed-mixed['upper'],sampled_single-single['upper']),
            center_source_loss=source_loss,center_radius=float(np.linalg.norm(v)),
            mixed_seconds=mixed_seconds,single_seconds=single_seconds,
            inaccurate_subproblems=sum(status!='optimal' for status in statuses),warnings=len(caught),
            solver_retries=sum(h['solver_retry'] for cell in single['cell_results'] for h in cell['history']),
            maximum_bound_inversion=max([h['bound_inversion'] for cell in single['cell_results'] for h in cell['history']]+[0.]),
            maximum_raw_bound_inversion=max([h['raw_bound_inversion'] for cell in single['cell_results'] for h in cell['history']]+[0.]),
            predicts_strict_gain=bound['certifies_strict_gain'] if bound else False,
            predicted_radius_gap=bound['radius_gap_lower'] if bound else '',
            predicted_single_lower=bound['single_radius_lower']**2 if bound else '',
            predicted_mixed_upper=bound['mixed_radius_upper']**2 if bound else '')
        cases.append(row)
        details[name]=dict(mixed_center=mixed['center'].tolist(),single_center=single['center'].tolist(),
            single_parameter=v.tolist(),bound=bound,mixed_history=mixed['history'],
            single_cell_histories=[c['history'] for c in single['cell_results']],warnings=[str(w.message) for w in caught])
        write_csv(out/'cases.csv',cases)
        print(name,'single gap',f"{row['single_gap']:.3g}",'price',f"{row['price_lower']:.5g}",flush=True)

    angle=np.arccos(cfg['base_angle_cosine']);baseH=np.array([[0.,-1.],[-np.sin(angle),np.cos(angle)]])
    baseR=rotation(.27);n=baseR@np.array([np.sqrt(2),1.])/np.sqrt(3)
    d=baseR@np.array([1.,-np.sqrt(2)])/np.sqrt(3)
    nominal_source=[negative_halfline_response(n),negative_halfline_response(-n)]
    nominal_target=[cone_projection_response(baseH@baseR.T)]
    P0=8/27;R0=1/8;hoffman=np.sqrt(2)
    for budget in cfg['source_loss_budgets']:
        bound=mixture_budget_bound(P0,R0,hoffman,1.,budget)
        run_case(f'budget_{budget:g}','budget',nominal_source,nominal_target,budget,bound,[d,-d])
    for source_angle in cfg['source_rotation_offsets']:
        for target_angle in cfg['target_rotation_offsets']:
            actual_n=rotation(source_angle)@n;actual_d=rotation(source_angle)@d
            source=[negative_halfline_response(actual_n),negative_halfline_response(-actual_n)]
            target=[cone_projection_response(baseH@(rotation(target_angle)@baseR).T)]
            delta=np.sqrt(2)*r*np.sin(abs(source_angle)/2);xi=4*r*np.sin(abs(target_angle)/2)
            budget=max(cfg['perturbation_minimum_budget'],delta*delta/2)
            bound=mixture_budget_bound(P0,R0,hoffman,1.,budget,delta,xi)
            run_case(f'perturb_{source_angle:g}_{target_angle:g}','perturbation',source,target,budget,bound,[actual_d,-actual_d])
    old=json.loads((ROOT/'results'/args.coupled_source/'metadata.json').read_text())
    for item in old['configurations']:
        if item['seed'] not in cfg['coupled_configuration_seeds']:continue
        source=[cone_projection_response(s['H'],s['L']) for s in item['source']]
        for kind in cfg['coupled_target_types']:
            target=([linear_response(L) for L in item['target_latent']] if kind=='released' else
                    [cone_projection_response(H,L) for H,L in zip(item['target_normals'],item['target_latent'])])
            for budget in cfg['coupled_budgets']:
                run_case(f"coupled_{item['seed']}_{kind}_{budget:g}",kind,source,target,budget)
    write_csv(out/'cases.csv',cases)
    metadata=dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        phase_cases=len(phase),projection_realizations=len(embedding),optimization_cases=len(cases),details=details,
        interpretation=cfg['interpretation'])
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('All cases retained:',len(cases),flush=True)


if __name__=='__main__':main()
