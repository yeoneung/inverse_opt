"""Coupled weak-face recovery, exact planar separation, and analytic controls."""
import csv
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.optimal_recovery import (cone_projection_response,negative_halfline_response,linear_response,
    common_cells,optimal_recovery,farthest_action,cone_sector_recovery)


def batch_response(response,points):
    output=np.zeros((len(points),response.regions[0].action.shape[0]));remaining=np.ones(len(points),bool)
    for region in response.regions:
        mask=remaining & np.all(points@region.rows.T>=-1e-11,axis=1)
        output[mask]=points[mask]@region.action.T;remaining[mask]=False
    if remaining.any():raise ValueError('Response cover incomplete')
    return output


def direct_grid(source,target,radius,budget,size):
    angles=np.linspace(0,2*np.pi,size,endpoint=False);V=np.column_stack((np.cos(angles),np.sin(angles)))
    energy=sum(np.sum(batch_response(s,V)**2,axis=1) for s in source)/len(source)
    if budget==0:length=np.where(energy<1e-20,radius,0.)
    else:length=np.minimum(radius,np.sqrt(2*budget/np.maximum(energy,1e-30)))
    parameters=np.vstack((np.zeros((1,2)),V*length[:,None]))
    actions=np.hstack([batch_response(t,parameters) for t in target])/np.sqrt(len(target))
    return parameters,actions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='recovery');args=parser.parse_args()
    if Path(args.output).name!=args.output:raise ValueError('Use a simple output name')
    p=ROOT/'configs/recovery_protocol.json';cfg=json.loads(p.read_text())
    out=ROOT/'results'/args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'runs.csv').exists():raise FileExistsError('Preserve first run')
    r=cfg['radius'];base=np.array(cfg['base_normals']);rows=[];configs=[];histories={};analytic=[]
    for seed in cfg['configuration_seeds']:
        rng=np.random.default_rng(seed);source=[];source_data=[]
        for _ in range(cfg['source_contexts']):
            H=base+cfg['source_normal_perturbation']*rng.normal(size=(2,2))
            L=np.eye(2)+cfg['source_latent_perturbation']*rng.normal(size=(2,2))
            source.append(cone_projection_response(H,L));source_data.append(dict(H=H.tolist(),L=L.tolist()))
        latent=[np.eye(2)+cfg['target_latent_perturbation']*rng.normal(size=(2,2)) for _ in range(cfg['target_contexts'])]
        target_normals=[]
        for _ in latent:
            angle=rng.uniform(*cfg['target_rotation_range']);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            target_normals.append(base@R)
        configs.append(dict(seed=seed,source=source_data,target_latent=[L.tolist() for L in latent],
                            target_normals=[H.tolist() for H in target_normals]))
        for kind in cfg['target_types']:
            target=([linear_response(L) for L in latent] if kind=='released' else
                    [cone_projection_response(H,L) for H,L in zip(target_normals,latent)])
            cells=common_cells(source,target)
            for budget in cfg['loss_budgets']:
                start=time.perf_counter();result=optimal_recovery(cells,r,budget,cfg['exchange_absolute_squared_gap'],cfg['exchange_max_iterations'])
                seconds=time.perf_counter()-start
                reference=farthest_action(cells,np.zeros(cells[0].action.shape[0]),r,budget)['value']
                parameters,actions=direct_grid(source,target,r,budget,cfg['independent_angular_grid'])
                sampled=float(np.max(np.sum((actions-result['center'])**2,axis=1)))
                witness=result['parameters']
                witness_losses=sum(np.sum(batch_response(s,witness)**2,axis=1) for s in source)/(2*len(source))
                key=f'{seed}_{kind}_{budget}';histories[key]=result['history']
                rows.append(dict(seed=seed,target_type=kind,budget=budget,cells=len(cells),
                    converged=result['converged'],lower=result['lower'],upper=result['upper'],
                    gap=result['upper']-result['lower'],reference_risk=reference,
                    upper_over_reference=result['upper']/reference if reference>1e-14 else 0.,
                    center_norm=float(np.linalg.norm(result['center'])),iterations=len(result['history']),
                    support_points=len(result['points']),max_witness_loss=float(witness_losses.max()),
                    max_witness_radius=float(np.linalg.norm(witness,axis=1).max()),
                    grid_worst_risk=sampled,grid_above_oracle=max(0.,sampled-result['upper']),seconds=seconds))
        print(f'configuration {seed} complete',flush=True)
    for phi in np.linspace(.05,1.5,24):
        source=[negative_halfline_response([np.sin(phi),np.cos(phi)]),negative_halfline_response([np.sin(phi),-np.cos(phi)])]
        cells=common_cells(source,[linear_response(np.eye(2))]);exact=cone_sector_recovery(np.cos(phi),[1.,0.])
        result=optimal_recovery(cells,1.,0.,1e-8)
        analytic.append(dict(half_angle=float(phi),exact_risk=exact['risk'],computed_lower=result['lower'],computed_upper=result['upper'],
            exact_center=exact['center'].tolist(),computed_center=result['center'].tolist(),converged=result['converged']))
    with (out/'runs.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    metadata=dict(protocol_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        configurations=configs,cases=len(rows),histories=histories,analytic_wedges=analytic,
        note='All outputs are floating-point diagnostics for exact-arithmetic theorems; no interval proof, statistical generalization or control-performance claim.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


if __name__=='__main__':main()
