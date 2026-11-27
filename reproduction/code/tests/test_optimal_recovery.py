"""Independent analytic and global-grid checks of the planar recovery oracle."""
import unittest
import numpy as np
from scipy.optimize import minimize
from src.optimal_recovery import (QuadraticCell,cone_projection_response,common_cells,
    negative_halfline_response,linear_response,farthest_action,optimal_recovery,
    sharp_transfer_constant_2d,cone_sector_recovery,scalar_cone_recovery,isometric_cone_from_rays)


class OptimalRecoveryTests(unittest.TestCase):
    def test_iteration_limit_preserves_a_realizable_center_and_bounds(self):
        cells=common_cells([negative_halfline_response([1.,0.])],
                           [linear_response([[1.,0.]])])
        result=optimal_recovery(cells,1.,0.,1e-9,max_iterations=1)
        self.assertFalse(result['converged'])
        np.testing.assert_allclose(result['weights']@result['points'],result['center'])
        np.testing.assert_allclose(result['points'],result['parameters']@cells[0].action.T)
        self.assertLessEqual(result['lower'],.25)
        self.assertGreaterEqual(result['upper'],.25)

    def test_scalar_formula_in_higher_dimension(self):
        result=scalar_cone_recovery(np.eye(3),[2.,-1.,0.],2.)
        self.assertAlmostEqual(result['center'],1.,places=7)
        self.assertAlmostEqual(result['risk'],9.,places=7)

    def test_finite_ray_reduction_in_higher_dimension(self):
        for p in [2,3,6,12]:
            result=isometric_cone_from_rays(np.eye(p))
            np.testing.assert_allclose(result['center'],np.ones(p)/p,atol=1e-8)
            self.assertAlmostEqual(result['risk'],1-1/p,places=8)
        rng=np.random.default_rng(990);rays=rng.normal(size=(8,5));rays[:,0]=np.abs(rays[:,0])+1.
        result=isometric_cone_from_rays(rays)
        weights=rng.uniform(size=(3000,len(rays)));points=weights@rays;points/=np.linalg.norm(points,axis=1)[:,None]
        self.assertLessEqual(np.max(np.sum((points-result['center'])**2,axis=1)),result['upper']+1e-8)
    def test_coupled_projection_matches_independent_qp(self):
        rng=np.random.default_rng(341);H=np.array([[1.,.45],[-.25,1.]])
        L=np.array([[1.2,.3],[-.4,.8]]);response=cone_projection_response(H,L)
        for _ in range(30):
            theta=rng.normal(size=2);z=L@theta
            solve=minimize(lambda u:.5*np.sum((u-z)**2),np.zeros(2),jac=lambda u:u-z,
                constraints={'type':'ineq','fun':lambda u:-H@u,'jac':lambda u:-H},
                method='SLSQP',options={'ftol':1e-12,'maxiter':200})
            self.assertTrue(solve.success)
            np.testing.assert_allclose(response(theta),solve.x,atol=2e-7)

    def test_common_cells_reproduce_both_responses(self):
        source=[cone_projection_response([[1.,.4],[-.2,1.]]),negative_halfline_response([1.,-.5])]
        target=[cone_projection_response([[.8,-.3],[.2,1.]],[[1.,.2],[-.1,.9]])]
        cells=common_cells(source,target);rng=np.random.default_rng(244)
        for _ in range(500):
            v=rng.normal(size=2);cell=next(c for c in cells if np.all(c.rows@v>=-1e-10))
            truth=sum(np.sum(s(v)**2) for s in source)/len(source)
            self.assertAlmostEqual(v@cell.information@v,truth,places=9)
            np.testing.assert_allclose(cell.action@v,target[0](v),atol=1e-9)

    def test_one_sided_ray_has_quarter_radius_risk(self):
        source=[negative_halfline_response([1.,0.]),negative_halfline_response([0.,1.]),
                negative_halfline_response([0.,-1.])]
        cells=common_cells(source,[linear_response([[1.,0.]])])
        result=optimal_recovery(cells,1.,0.,1e-9)
        self.assertTrue(result['converged']);self.assertAlmostEqual(result['upper'],.25,places=7)
        np.testing.assert_allclose(result['center'],[.5],atol=1e-7)

    def test_quarter_disk_center_and_risk(self):
        source=[negative_halfline_response([1.,0.]),negative_halfline_response([0.,1.])]
        cells=common_cells(source,[linear_response(np.eye(2))])
        result=optimal_recovery(cells,1.,0.,1e-8)
        self.assertTrue(result['converged']);self.assertAlmostEqual(result['upper'],.5,places=6)
        np.testing.assert_allclose(result['center'],[.5,.5],atol=1e-5)

    def test_target_mixture_can_outperform_every_single_parameter(self):
        source=[negative_halfline_response([0.,1.]),negative_halfline_response([0.,-1.])]
        target=[negative_halfline_response([1.,0.]),negative_halfline_response([-1.,0.])]
        cells=common_cells(source,target);result=optimal_recovery(cells,1.,0.,1e-9)
        self.assertTrue(result['converged'])
        np.testing.assert_allclose(result['center'],[-1/(2*np.sqrt(2))]*2,atol=1e-7)
        self.assertAlmostEqual(result['upper'],.25,places=7)
        # Any image point lies on one axis, and its error to the other endpoint is >= .5.
        self.assertAlmostEqual(farthest_action(cells,np.zeros(2),1.,0.)['value'],.5,places=8)

    def test_planar_oracle_matches_dense_angular_global_check(self):
        rng=np.random.default_rng(658)
        theta=np.linspace(0,2*np.pi,18001);directions=np.column_stack((np.cos(theta),np.sin(theta)))
        for I in [np.array([[1.2,.3],[.3,.6]]),np.array([[1.,.2],[.2,.04]]),np.zeros((2,2))]:
            for budget in [0.,.03,.4]:
                A=rng.normal(size=(3,2));center=rng.normal(size=3)
                cell=QuadraticCell(np.empty((0,2)),I,A)
                result=farthest_action([cell],center,1.,budget)
                energy=np.einsum('ni,ij,nj->n',directions,I,directions)
                lengths=np.minimum(1.,np.sqrt(2*budget/np.maximum(energy,1e-30)))
                if np.linalg.norm(I)==0:lengths[:]=1.
                sampled=(directions*lengths[:,None])@A.T
                lower=max(np.sum(center**2),float(np.max(np.sum((sampled-center)**2,axis=1))))
                self.assertGreaterEqual(result['value']+1e-6,lower)
                # At zero budget a rank-one null direction may be missed by the dense grid.
                if budget>0 or np.linalg.matrix_rank(I)!=1:
                    self.assertLess(result['value']-lower,3e-3)

    def test_sharp_constant_and_positive_budget_match_isotropic_formula(self):
        source=[negative_halfline_response(v) for v in [[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]]
        cells=common_cells(source,[linear_response(np.eye(2))])
        result=sharp_transfer_constant_2d(cells)
        self.assertAlmostEqual(result['value'],4.,places=8)
        recovered=optimal_recovery(cells,1.,.03125,1e-9)
        self.assertTrue(recovered['converged']);self.assertAlmostEqual(recovered['upper'],.25,places=7)
        np.testing.assert_allclose(recovered['center'],np.zeros(2),atol=1e-7)

    def test_isometric_cone_formula_agrees_with_wedge_oracle(self):
        for phi in [.1,.45,.9,1.3]:
            rows=np.array([[np.sin(phi),np.cos(phi)],[np.sin(phi),-np.cos(phi)]])
            cell=QuadraticCell(rows,np.zeros((2,2)),np.eye(2))
            explicit=cone_sector_recovery(np.cos(phi),[1.,0.])
            result=optimal_recovery([cell],1.,0.,1e-8)
            self.assertTrue(result['converged']);self.assertAlmostEqual(result['upper'],explicit['risk'],places=6)
            np.testing.assert_allclose(result['center'],explicit['center'],atol=5e-5)

    def test_sharp_piecewise_modulus_against_direct_direction_grid(self):
        rng=np.random.default_rng(971)
        angles=np.linspace(0,2*np.pi,14001);V=np.column_stack((np.cos(angles),np.sin(angles)))
        for _ in range(12):
            source_rows=np.array([[1.,.1],[.15,1.],[.8,.7]])+.04*rng.normal(size=(3,2))
            target_rows=rng.uniform(.1,1.,(2,3))@source_rows
            source=[negative_halfline_response(row) for row in source_rows]
            target=[negative_halfline_response(row) for row in target_rows]
            actual=sharp_transfer_constant_2d(common_cells(source,target))['value']
            source_energy=np.mean(np.minimum(V@source_rows.T,0.)**2,axis=1)
            target_energy=np.mean(np.minimum(V@target_rows.T,0.)**2,axis=1)
            visible=source_energy>1e-10;lower=np.max(target_energy[visible]/source_energy[visible])
            self.assertTrue(np.isfinite(actual));self.assertGreaterEqual(actual+1e-7,lower)
            self.assertLess(actual-lower,2e-4)
        exposed=common_cells([negative_halfline_response([1.,0.])],[linear_response([[1.,0.]])])
        self.assertTrue(np.isinf(sharp_transfer_constant_2d(exposed)['value']))


if __name__=='__main__':unittest.main()
