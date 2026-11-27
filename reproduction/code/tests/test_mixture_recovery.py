"""Independent optimization checks of mixture necessity and persistence."""
import unittest
import numpy as np
from scipy.optimize import minimize, minimize_scalar, nnls
from src.optimal_recovery import (common_cells, negative_halfline_response,
    cone_projection_response, linear_response, optimal_recovery, farthest_action)
from src.mixture_recovery import (two_branch_recovery, consistent_model_recovery,
    mixture_budget_bound)


class MixtureRecoveryTests(unittest.TestCase):
    def test_two_branch_formula_against_independent_primal_optimization(self):
        rng=np.random.default_rng(5241)
        for _ in range(30):
            a=rng.normal(size=4);b=rng.normal(size=4)*rng.uniform(.1,2.)
            result=two_branch_recovery(a,b);points=np.vstack((np.zeros(4),a,b))
            def residual(z):return z[-1]-np.sum((points-z[:-1])**2,axis=1)
            solved=minimize(lambda z:z[-1],np.r_[np.zeros(4),np.max(np.sum(points**2,axis=1))],
                constraints={'type':'ineq','fun':residual},method='SLSQP',
                options={'ftol':1e-11,'maxiter':500})
            self.assertTrue(solved.success)
            self.assertAlmostEqual(result['mixed_risk'],solved.fun,places=6)
            values=[]
            for endpoint in [a,b]:
                f=lambda t:float(np.max(np.sum((points-t*endpoint)**2,axis=1)))
                optimum=minimize_scalar(f,bounds=(0.,1.),method='bounded',options={'xatol':1e-12})
                values.append(min(f(0.),f(1.),optimum.fun))
            self.assertAlmostEqual(result['single_risk'],min(values),places=6)

    def test_sharp_constant_is_realized_by_one_quadratic_projection(self):
        a=np.array([1.,0.]);b=np.array([1/3,2*np.sqrt(2)/3]);d=np.array([1.,-np.sqrt(2)])
        G=np.column_stack((a,b))
        for sign,truth in [(1.,a),(-1.,b)]:
            coefficients,_=nnls(G,sign*d)
            np.testing.assert_allclose(G@coefficients,truth,atol=1e-10)
        result=two_branch_recovery(a,b)
        self.assertAlmostEqual(result['mixed_risk'],3/8,places=10)
        self.assertAlmostEqual(result['single_risk'],8/9,places=10)
        self.assertAlmostEqual(result['ratio'],64/27,places=10)
        self.assertTrue(result['strict_gain'])
        np.testing.assert_allclose(result['endpoint_weights']@np.vstack((np.zeros(2),a,b)),result['mixed_center'])

    def test_switching_does_not_always_require_mixing(self):
        a=np.array([1.,0.]);b=.5*np.array([.7,np.sqrt(1-.7**2)])
        result=two_branch_recovery(a,b)
        self.assertFalse(result['strict_gain'])
        self.assertAlmostEqual(result['mixed_risk'],result['single_risk'],places=12)
        np.testing.assert_allclose(result['single_center'],a/2)

    def test_collinear_and_zero_branches_have_no_gap(self):
        for b in [[0.,0.],[.4,0.],[-.4,0.],[-1.,0.],[2.,0.]]:
            result=two_branch_recovery([1.,0.],b)
            self.assertFalse(result['strict_gain'])
            self.assertAlmostEqual(result['mixed_risk'],result['single_risk'],places=10)
        self.assertEqual(two_branch_recovery([0.],[0.])['mixed_risk'],0.)

    def test_global_ratio_bound_and_scaling(self):
        rng=np.random.default_rng(775)
        for _ in range(300):
            k=rng.uniform(.001,1.);rho=rng.uniform(-1.,1.)
            a=np.array([1.,0.]);b=k*np.array([rho,np.sqrt(1-rho*rho)])
            result=two_branch_recovery(a,b)
            self.assertLessEqual(result['ratio'],64/27+1e-10)
            self.assertGreaterEqual(result['ratio'],1-1e-10)
            scaled=two_branch_recovery(2.3*a,2.3*b)
            self.assertAlmostEqual(scaled['single_risk'],2.3**2*result['single_risk'],places=9)

    @staticmethod
    def coupled_model():
        theta=np.arccos(1/3);H=np.array([[0.,-1.],[-np.sin(theta),np.cos(theta)]])
        n=np.array([np.sqrt(2),1.])/np.sqrt(3)
        source=[negative_halfline_response(n),negative_halfline_response(-n)]
        return common_cells(source,[cone_projection_response(H)])

    def test_consistent_solver_and_budget_persistence(self):
        cells=self.coupled_model();P0=8/27;R0=1/8
        for budget in [0.,.001]:
            mixed=optimal_recovery(cells,1.,budget,1e-8)
            single=consistent_model_recovery(cells,1.,budget,initial_points=mixed['points'])
            self.assertTrue(single['converged'])
            self.assertTrue(mixed['converged'])
            bound=mixture_budget_bound(P0,R0,np.sqrt(2),1.,budget)
            self.assertTrue(bound['certifies_strict_gain'])
            self.assertGreaterEqual(np.sqrt(single['upper'])+1e-6,bound['single_radius_lower'])
            self.assertLessEqual(np.sqrt(mixed['upper']),bound['mixed_radius_upper']+1e-6)
            if budget==0:
                self.assertAlmostEqual(single['upper'],P0,places=6)
                self.assertAlmostEqual(mixed['upper'],R0,places=6)
        # At full fit budget the image is K intersected with the unit ball, hence convex.
        mixed=optimal_recovery(cells,1.,.25,1e-8)
        single=consistent_model_recovery(cells,1.,.25,initial_points=mixed['points'])
        self.assertAlmostEqual(single['upper'],mixed['upper'],places=6)
        self.assertFalse(mixture_budget_bound(P0,R0,np.sqrt(2),1.,.25)['certifies_strict_gain'])

    def test_linear_targets_need_no_mixture_at_positive_budget(self):
        source=[negative_halfline_response([1.,.3]),negative_halfline_response([-1.,-.3])]
        cells=common_cells(source,[linear_response([[1.2,.4],[-.2,.9]])])
        mixed=optimal_recovery(cells,1.,.02,1e-8)
        single=consistent_model_recovery(cells,1.,.02,initial_points=mixed['points'])
        self.assertTrue(single['converged'])
        self.assertAlmostEqual(single['upper'],mixed['upper'],places=5)
        self.assertLessEqual(single['lower'],mixed['upper']+1e-6)

    def test_perturbation_bound_requires_nominal_class_inclusion(self):
        with self.assertRaises(ValueError):
            mixture_budget_bound(1.,.5,1.,1.,0.,source_error=.01)
        result=mixture_budget_bound(1.,.4,1.,1.,.0001,source_error=.005,target_error=.002)
        self.assertTrue(result['certifies_strict_gain'])


if __name__=='__main__':unittest.main()
