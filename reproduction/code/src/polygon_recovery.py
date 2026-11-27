"""Exact rational recovery on polygonal projection images.

The normal-ray partition verifies that the complete boundary arc is the image
of a straight latent segment. Radius and restricted-center calculations then
use rational arithmetic, rather than a grid approximation.
"""
from fractions import Fraction as F
from itertools import combinations
import numpy as np


def add(a,b):return tuple(x+y for x,y in zip(a,b))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def scale(t,a):return tuple(t*x for x in a)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
def vec(a):return tuple(F(x) for x in a)


def rational_instances():
    return [dict(name='quadrilateral',vertices=list(map(vec,[(0,0),(1,0),('117/100','47/100'),('81/200','111/100')])),
                 offset=vec((6,-2)),direction=vec((1,3))),
            dict(name='pentagon',vertices=list(map(vec,[(0,0),('7/8',0),('71/50','18/25'),('53/50','37/25'),('3/50','25/16')])),
                 offset=vec((3,-1)),direction=vec((1,24)))]


def normal_ray_partition(vertices,offset,direction):
    vertices=list(map(vec,vertices));offset=vec(offset);direction=vec(direction)
    cyclic=[sub(vertices[(i+1)%len(vertices)],v) for i,v in enumerate(vertices)]
    if not all(cross(cyclic[i],cyclic[(i+1)%len(cyclic)])>0 for i in range(len(cyclic))):
        raise ValueError('Vertices must form the specified strictly convex CCW polygon')
    if not all(cross(edge,sub(point,start))>=0 for start,edge in zip(vertices,cyclic) for point in vertices):
        raise ValueError('Every vertex must satisfy all polygon edge halfspaces')
    intervals=[]
    for start,end in zip(vertices[:-1],vertices[1:]):
        edge=sub(end,start);normal=(edge[1],-edge[0]);minus_normal=scale(-1,normal)
        determinant=cross(direction,minus_normal)
        if not determinant:raise ValueError('Latent line parallel to an edge normal')
        hits=[]
        for vertex in [start,end]:
            residual=sub(vertex,offset)
            t=cross(residual,minus_normal)/determinant
            multiplier=cross(direction,residual)/determinant
            if multiplier<0:raise ValueError('A normal ray is reached on its negative extension')
            assert add(offset,scale(t,direction))==add(vertex,scale(multiplier,normal))
            hits.append((t,multiplier))
        if hits[0][0]>hits[1][0]:raise ValueError('Projection reverses edge order')
        if intervals and intervals[-1]['end']>hits[0][0]:
            raise ValueError('Normal sectors do not have the stated order')
        intervals.append(dict(start=hits[0][0],end=hits[1][0],multipliers=[h[1] for h in hits]))
    return intervals


def enclosing_radius(vertices):
    vertices=list(map(vec,vertices));candidates=[]
    for a,b in combinations(vertices,2):candidates.append(scale(F(1,2),add(a,b)))
    for a,b,c in combinations(vertices,3):
        rows=[scale(2,sub(b,a)),scale(2,sub(c,a))]
        determinant=cross(*rows)
        if not determinant:continue
        rhs=[dot(b,b)-dot(a,a),dot(c,c)-dot(a,a)]
        candidates.append(((rhs[0]*rows[1][1]-rows[0][1]*rhs[1])/determinant,
                           (rows[0][0]*rhs[1]-rhs[0]*rows[1][0])/determinant))
    if not candidates:return F(0),vertices[0]
    return min((max(dot(sub(y,x),sub(y,x)) for y in vertices),x) for x in candidates)


def restricted_radius(vertices):
    vertices=list(map(vec,vertices));best=None
    for edge_index,(start,end) in enumerate(zip(vertices[:-1],vertices[1:])):
        edge=sub(end,start);quadratic=dot(edge,edge)
        if not quadratic:continue
        linear=[2*dot(edge,sub(start,y)) for y in vertices]
        constant=[dot(sub(start,y),sub(start,y)) for y in vertices]
        candidates={F(0),F(1)}
        candidates.update(-b/(2*quadratic) for b in linear)
        for j,k in combinations(range(len(vertices)),2):
            if linear[j]!=linear[k]:
                candidates.add((constant[k]-constant[j])/(linear[j]-linear[k]))
        for t in candidates:
            if not 0<=t<=1:continue
            value=quadratic*t*t+max(b*t+c for b,c in zip(linear,constant))
            entry=(value,add(start,scale(t,edge)),edge_index,t)
            if best is None or value<best[0]:best=entry
    return best


def exact_case(instance):
    intervals=normal_ray_partition(instance['vertices'],instance['offset'],instance['direction'])
    mixed,center=enclosing_radius(instance['vertices'])
    proper,single,edge,t=restricted_radius(instance['vertices'])
    assert mixed>0 and proper<=4*mixed
    return dict(name=instance['name'],mixed=mixed,proper=proper,ratio=proper/mixed,
                mixed_center=center,single_center=single,single_edge=edge,single_fraction=t,
                parameter_interval=(intervals[0]['start'],intervals[-1]['end']),partition=intervals)


def project_polygon(points,vertices):
    """Independent floating-point projections onto the full polygon."""
    points=np.atleast_2d(np.asarray(points,dtype=float));vertices=np.asarray(vertices,dtype=float)
    edges=np.roll(vertices,-1,axis=0)-vertices
    normals=np.column_stack([edges[:,1],-edges[:,0]])
    inside=np.max(np.sum((points[:,None]-vertices)*normals,axis=2),axis=1)<=1e-12
    out=points.copy();best=np.full(len(points),np.inf)
    for start,edge in zip(vertices,edges):
        t=np.clip((points-start)@edge/(edge@edge),0.,1.)
        projected=start+t[:,None]*edge
        distance=np.sum((points-projected)**2,axis=1)
        update=(distance<best)&~inside
        out[update]=projected[update];best[update]=distance[update]
    return out
