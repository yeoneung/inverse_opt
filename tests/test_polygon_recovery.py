"""Checks of global image complexity and the centered-facet restriction."""
import unittest
from fractions import Fraction as F
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar
from src.polygon_recovery import rational_instances,exact_case,project_polygon,dot,sub
from experiments.classification import enclosing


class PolygonRecoveryTests(unittest.TestCase):
    def test_global_ratios_exceed_two_branch_bound_with_exact_arithmetic(self):
        for instance in rational_instances():
            result=exact_case(instance)
            self.assertGreater(result['ratio'],F(64,27));self.assertLessEqual(result['ratio'],4)
            if instance['name']=='pentagon':self.assertGreater(result['ratio'],3)
            center=result['mixed_center']
            self.assertTrue(all(dot(sub(y,center),sub(y,center))<=result['mixed'] for y in instance['vertices']))

    def test_exact_normal_fan_partition_matches_full_polygon_projection(self):
        for instance in rational_instances():
            result=exact_case(instance);vertices=np.array(instance['vertices'],dtype=float)
            offset=np.array(instance['offset'],dtype=float);direction=np.array(instance['direction'],dtype=float)
            for index,interval in enumerate(result['partition']):
                t=np.linspace(float(interval['start']),float(interval['end']),9)
                actual=project_polygon(offset+t[:,None]*direction,vertices)
                expected=vertices[index]+np.linspace(0,1,9)[:,None]*(vertices[index+1]-vertices[index])
                self.assertLess(np.max(np.linalg.norm(actual-expected,axis=1)),1e-10)

    def test_rational_radii_match_independent_conic_centers(self):
        for instance in rational_instances():
            result=exact_case(instance);vertices=np.array(instance['vertices'],dtype=float)
            mixed,_=enclosing(vertices)
            proper=min(enclosing(vertices,[a,b])[0] for a,b in zip(vertices[:-1],vertices[1:]))
            self.assertAlmostEqual(mixed,float(result['mixed']),places=7)
            self.assertAlmostEqual(proper,float(result['proper']),places=7)

    def test_off_center_single_facet_can_approach_two(self):
        for beta in [.1,.01,.0001]:
            d=np.array([np.sqrt(beta),np.sqrt(1-beta)])
            left=np.array([-1.,0.]);right=d
            points=np.array([left,np.zeros(2),right])
            mixed,_=enclosing(points)
            proper=min(enclosing(points,[left,np.zeros(2)])[0],enclosing(points,[np.zeros(2),right])[0])
            self.assertAlmostEqual(proper/mixed,2/(1+np.sqrt(beta)),places=7)

    def test_shared_capacity_example_uses_actual_coupled_projection(self):
        x0=np.ones(2)/2;n=-np.ones(2)/np.sqrt(2);e=np.array([1.,-1.])/np.sqrt(2)
        d=e/np.sqrt(3)+np.sqrt(2/3)*n;r=.3
        z=cp.Parameter(2);x=cp.Variable(2)
        problem=cp.Problem(cp.Minimize(cp.sum_squares(x-z)),[x>=0,cp.sum(x)<=1])
        for t in np.linspace(-r,r,13):
            z.value=x0+t*d;problem.solve(solver='CLARABEL',tol_gap_abs=1e-14,tol_gap_rel=1e-14,tol_feas=1e-13,max_iter=300)
            expected=x0+t*(d if t>=0 else e/np.sqrt(3))
            self.assertLess(np.linalg.norm(x.value-expected),1e-7)
        endpoints=np.array([x0-r*e/np.sqrt(3),x0,x0+r*d])
        mixed,_=enclosing(endpoints)
        proper=min(enclosing(endpoints,endpoints[:2])[0],enclosing(endpoints,endpoints[1:])[0])
        self.assertAlmostEqual(mixed,r*r/2,places=7)
        self.assertAlmostEqual(proper,9*r*r/16,places=7)
        # Independently solve the fixed-fibre separation: the mixed center is
        # interior, so its normal cone is zero. Source consistency is H_d v=0.
        mixed_center=x0+r*n/np.sqrt(6);H_d=np.eye(2)-np.outer(d,d)
        preference=cp.Variable(2)
        fibre=cp.Problem(cp.Minimize(cp.norm(x0+preference-mixed_center)),
                         [H_d@preference==0,cp.norm(preference)<=r])
        fibre.solve(solver='CLARABEL',tol_gap_abs=1e-12,tol_gap_rel=1e-12,tol_feas=1e-12)
        margin=min(mixed_center.min(),(1-mixed_center.sum())/np.sqrt(2))
        self.assertAlmostEqual(margin,r/np.sqrt(6),places=10)
        self.assertAlmostEqual(fibre.value,r/np.sqrt(18),places=9)
        self.assertLess(fibre.value,margin)
        self.assertLess(np.linalg.norm(preference.value-r*d/3),1e-7)
        self.assertAlmostEqual(fibre.value**2/(proper-mixed),8/9,places=7)
        nearest=x0+preference.value
        self.assertAlmostEqual(np.max(np.sum((endpoints-nearest)**2,axis=1)),2*r*r/3,places=7)


if __name__=='__main__':unittest.main()
