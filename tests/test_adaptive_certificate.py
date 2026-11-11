"""Image-distance calibration, equality normals and radius-search checks."""
import unittest
import numpy as np
from src.adaptive_certificate import adaptive_certificate
from src.projection_paths import mixed_radius,polyhedral_line_path
from experiments.capacity import capacity_instance


class AdaptiveCertificateTests(unittest.TestCase):
    def test_unbounded_target_improves_original_margin(self):
        result=adaptive_certificate([1.,0.],[1.,2.],[[-1.,0.]],[0.],[1.,0.],interval=(-1.,1.))
        self.assertAlmostEqual(result['lower'],2.,places=6)

    def test_boundary_anchor_and_tied_slacks(self):
        result=adaptive_certificate([0.,1.],[-1.,1.],-np.eye(2),[0.,0.],[0.,0.],interval=(0.,1.))
        self.assertAlmostEqual(result['lower'],1.,places=6)
        result=adaptive_certificate([1.,-1.],[0.,0.],-np.eye(2),[0.,0.],[.5,.5],interval=(-1.,1.))
        self.assertAlmostEqual(result['lower'],.5,places=7)

    def test_affine_target_and_equality_normal_are_supported(self):
        result=adaptive_certificate([1.,0.],[0.,1.],np.empty((0,2)),[],[0.,0.],
                                    interval=(-1.,1.),equalities=[[0.,1.]])
        self.assertLess(result['lower'],1e-7)
        result=adaptive_certificate([1.,0.],[0.,2.],np.empty((0,2)),[],[0.,0.],interval=(-1.,1.))
        self.assertAlmostEqual(result['lower'],2.,places=6)

    def test_ball_prior_and_homogeneous_sources_have_correct_normalization(self):
        x=np.ones(2)*.8
        source=[(np.empty((0,2)),np.eye(2))]
        result=adaptive_certificate(np.eye(2),np.zeros(2),-np.eye(2),[0.,0.],x,
                                    radius=1.,sources=source,budget=.125)
        self.assertAlmostEqual(result['lower'],np.linalg.norm(x)-.5,places=6)
        for d in [2,4,8]:
            G=np.vstack((np.eye(d),-np.eye(d)));anchor=np.ones(2*d)/(2*d)
            result=adaptive_certificate(G,np.zeros(2*d),-np.eye(2*d),np.zeros(2*d),anchor,radius=1.)
            self.assertAlmostEqual(result['lower'],1/(2*d),places=7)

    def test_binary_radius_search_matches_exhaustive_candidate_values(self):
        for q in [4,8,16,32]:
            A,b,c,d,left,right=capacity_instance(q,552000,.8)
            vertices,_,_,_=polyhedral_line_path(A,b,c,d,left,right)
            _,_,x=mixed_radius(vertices)
            first=adaptive_certificate(d,c,A,b,x,interval=(left,right))
            all_values=adaptive_certificate(d,c,A,b,x,interval=(left,right),exhaustive=True)
            self.assertAlmostEqual(first['lower'],all_values['lower'],places=6)
            self.assertLessEqual(first['solves'],int(np.ceil(np.log2(first['thresholds'])))+2)


if __name__=='__main__':unittest.main()
