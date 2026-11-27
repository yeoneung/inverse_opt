"""Independent geometry and convex-dual checks of the all-dimensional fibre test."""
import unittest
import numpy as np
from scipy.optimize import minimize_scalar
from src.fiber_certificate import fiber_certificate,finite_witness_gap,global_fiber_primal


class FiberCertificateTests(unittest.TestCase):
    def test_global_sandwich_and_exact_local_distance(self):
        a=np.array([.8,.8]);result=fiber_certificate(np.eye(2),np.eye(2),a)
        expected=np.linalg.norm(a)-1.
        self.assertTrue(result['local_identity_region'])
        self.assertAlmostEqual(result['upper'],expected,places=7)
        self.assertLessEqual(result['lower'],expected+1e-8)
        self.assertLess(expected-result['lower'],1e-6)
        # Beyond the face margin, a folded target gives a strict upper discrepancy.
        a=np.ones(2)/2;result=fiber_certificate([[1.],[-1.]],np.eye(2),a)
        solved=minimize_scalar(lambda x:float(np.sum((np.maximum([x,-x],0)-a)**2)),
                              bounds=(0.,1.),method='bounded')
        self.assertAlmostEqual(np.sqrt(solved.fun),.5,places=7)
        self.assertAlmostEqual(result['upper'],1/np.sqrt(2),places=7)
        self.assertLessEqual(result['distance_lower'],np.sqrt(solved.fun)+1e-8)
        # Sharp quadratic onset for a nonconvex affine projection image.
        for t in [0.,.1,.3,.6,1.]:
            c=(1-t)*np.ones(2)/2;L=(1+t)*np.array([[1.],[-1.]])/2
            actual=global_fiber_primal(L,c,np.eye(2),np.zeros(2),[.5,.5],[0.],1.,[])
            self.assertAlmostEqual(actual['upper'],t/np.sqrt(2),places=7)
            def error(v):
                x=np.maximum(c+L[:,0]*v,0.)
                return float(np.max(np.sum((np.eye(2)-x)**2,axis=1)))
            single=minimize_scalar(error,bounds=(-1.,1.),method='bounded',options={'xatol':1e-12})
            self.assertAlmostEqual(single.fun,.5*(1+t*t),places=7)

    def test_boundary_anchor_and_lower_dimensional_target(self):
        result=fiber_certificate([[0.],[1.]],np.eye(2),[1.,0.])
        self.assertAlmostEqual(result['upper'],1.,places=7)
        self.assertLessEqual(result['dual_feasibility_residual'],1e-7)
        # A ray embedded in R^2 exercises equality-normal directions.
        result=fiber_certificate(np.eye(2),[[1.,0.]],[.2,0.],target_equalities=[[0.,1.]])
        self.assertLess(result['upper'],1e-7)
        self.assertEqual(result['lower'],0.)
        # A radius-three prior crosses the bounded target's upper faces; this
        # checks the global statement beyond the original critical-cone ball.
        source=[]
        for sign in [1.,-1.]:
            source.append(dict(A=[[-1.],[1.]],b=[0.,1.],y=[0.],L=[[0.,sign]],c=[0.]))
        H=np.vstack((np.eye(2),-np.eye(2)));bounds=np.r_[np.zeros(2),-2*np.ones(2)]
        for budget in [0.,.01]:
            global_result=global_fiber_primal([[1.,0.],[-1.,0.]],[-1.,1.],H,bounds,
                                             [1.,1.],[1.,0.],3.,source,budget)
            self.assertAlmostEqual(global_result['upper'],np.sqrt(2),places=7)

    def test_positive_source_budget_dual_has_correct_normalization(self):
        # Source K=R^2 implies V is a ball of radius sqrt(2*budget).
        a=np.ones(2)/2
        result=fiber_certificate(np.eye(2),np.eye(2),a,budget=.125,
                                 sources=[(np.empty((0,2)),np.eye(2))])
        expected=np.linalg.norm(a)-.5
        self.assertAlmostEqual(result['upper'],expected,places=7)
        self.assertLess(abs(result['lower']-expected),1e-6)
        self.assertTrue(result['local_identity_region'])
        opposed=[(np.array([[-1.]]),np.array([[sign]])) for sign in [1.,-1.]]
        result=fiber_certificate([[1.]],np.empty((0,1)),[.8],budget=.01,sources=opposed)
        self.assertAlmostEqual(result['upper'],.6,places=7)
        self.assertLess(abs(result['lower']-.6),1e-6)
        nonzero_label=[dict(A=[[-1.],[1.]],b=[0.,1.],y=[1.],L=[[1.]],c=[0.])]
        result=global_fiber_primal([[1.]],[0.],np.empty((0,1)),[],[.25],[0.],2.,nonzero_label,.125)
        self.assertAlmostEqual(result['upper'],.25,places=7)

    def test_many_invisible_directions_and_finite_contact_witness(self):
        for dimension in [2,4,8]:
            L=np.vstack((np.eye(dimension),-np.eye(dimension)))
            a=np.ones(2*dimension)/(2*dimension)
            result=fiber_certificate(L,np.eye(2*dimension),a)
            self.assertAlmostEqual(result['upper'],1/np.sqrt(2*dimension),places=7)
            R=1-1/(2*dimension)
            # The variance bound needs witnesses in the actual response image,
            # not merely feasible target actions. Generate them from prior points.
            witness_parameters=np.vstack((np.eye(dimension),-np.eye(dimension)))
            witnesses=np.maximum(witness_parameters@L.T,0.)
            self.assertTrue(np.all(np.linalg.norm(witness_parameters,axis=1)<=1.))
            self.assertLess(np.linalg.norm(witnesses.mean(axis=0)-a),1e-14)
            variance=np.mean(np.sum((witnesses-a)**2,axis=1))
            self.assertAlmostEqual(variance,R,places=14)
            bound=finite_witness_gap(result,variance,R)
            self.assertTrue(bound['strict_gain'])
            self.assertAlmostEqual(bound['single_minus_mixed_lower'],1/(4*dimension**2),places=7)
            self.assertLessEqual(result['dual_feasibility_residual'],1e-7)
            # A nonlinear but convex target image has a realizable center.
            c=np.ones(dimension)/dimension
            control=fiber_certificate(np.eye(dimension),np.eye(dimension),c)
            self.assertLess(control['upper'],1e-7)
            self.assertEqual(control['lower'],0.)


if __name__=='__main__':unittest.main()
