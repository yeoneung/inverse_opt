"""Finite KKT multiplier bounds for nonnegative capacity-constrained projections.

For X={0<=y<=u, Dy<=d}, u>0, D>=0, d>0, all formulas are derived
from stationarity and complementarity. They do not use sampled multiplier
maxima or a guessed big-M. Floating-point padding is not interval arithmetic.
"""
from itertools import product
import time

import numpy as np


def capacity_multiplier_bounds(A,b,c,G,*,margin=1e-9,corner_dimension_limit=12):
    start=time.perf_counter()
    A=np.asarray(A,float);b=np.asarray(b,float);c=np.asarray(c,float);G=np.asarray(G,float)
    q,p=G.shape
    if A.shape[1]!=q or len(b)!=len(A) or len(A)<2*q:
        raise ValueError('Expected A=[-I;I;D], b=[0;u;d]')
    if not (np.array_equal(A[:q],-np.eye(q)) and np.array_equal(A[q:2*q],np.eye(q))
            and np.array_equal(b[:q],np.zeros(q))):
        raise ValueError('Capacity structure must hold exactly, without tolerance-based row identification')
    u=b[q:2*q];D=A[2*q:];d=b[2*q:]
    if np.any(u<=0) or np.any(D<0) or np.any(d<=0):
        raise ValueError('Bounds require u>0, D>=0, d>0')
    zmin=c-np.sum(np.abs(G),axis=1)
    zmax=c+np.sum(np.abs(G),axis=1)
    if p<=corner_dimension_limit:
        corners=np.asarray(list(product([-1.,1.],repeat=p)),float)
        preferred=c+corners@G.T
        quarter=float(np.max(.25*np.sum(np.maximum(preferred,0.)**2,axis=1)))
        ideal=np.clip(preferred/2,0.,u)
        clipped=float(np.max(np.sum(preferred*ideal-ideal**2,axis=1)))
        count=len(corners)
        method='all_latent_box_corners'
    else:
        quarter=float(.25*np.sum(np.maximum(zmax,0.)**2))
        ideal=np.clip(zmax/2,0.,u)
        clipped=float(np.sum(zmax*ideal-ideal**2))
        count=0
        method='componentwise_latent_box_upper'
    if clipped>quarter+1e-10*max(1.,quarter):
        raise RuntimeError('Clipped objective bound exceeds its unbounded-coordinate relaxation')
    C=min(quarter,clipped)
    C_padded=C+margin*max(1.,abs(C))
    resource=C_padded/d
    resource+=margin*np.maximum(1.,np.abs(resource))
    upper=np.maximum(zmax-u,0.)
    lower=np.maximum(D.T@resource-zmin,0.)
    bounds=np.r_[lower,upper,resource]
    bounds+=margin*np.maximum(1.,np.abs(bounds))
    return dict(lambda_upper=bounds,resource_multiplier_upper=resource,
        lower_box_multiplier_upper=bounds[:q],upper_box_multiplier_upper=bounds[q:2*q],
        preferred_lower=zmin,preferred_upper=zmax,stationarity_sum_upper=C_padded,
        quarter_positive_part_bound=quarter,action_clipped_bound=clipped,
        corner_count=count,method=method,seconds=time.perf_counter()-start,
        scope='Exact structural derivation with floating-point padding; no interval claim.')
