"""Analytic controls for finite-box support functions and normal separation."""
import unittest
import numpy as np
from src.box_certificate import box_cone_distance
from src.adaptive_certificate import radius_search


class BoxCertificateTests(unittest.TestCase):
    def test_asymmetric_box_uses_its_actual_support_function(self):
        result=box_cone_distance([[1.,2.]], [0.], np.empty((0,1)), [7.], [-1.,0.], [2.,1.])
        self.assertAlmostEqual(result['upper'],3.,places=7)
        self.assertAlmostEqual(result['lower'],3.,places=6)
        self.assertLess(result['dual_feasibility_residual'],1e-8)

    def test_multidimensional_fold_has_known_distance_and_penalty(self):
        p=2;gamma=.3
        G=np.vstack((np.eye(p),-np.eye(p)))*(1+gamma)/2
        x=np.full(2*p,.5);c=np.full(2*p,(1-gamma)/2)
        oracle=lambda rows:box_cone_distance(G,c,rows,x,-np.ones(p),np.ones(p))
        fixed=oracle(np.empty((0,2*p)))
        self.assertAlmostEqual(fixed['upper'],gamma*np.sqrt(p/2),places=7)
        adaptive=radius_search(-np.eye(2*p),np.zeros(2*p),x,oracle)
        self.assertAlmostEqual(adaptive['lower']**2,p*gamma**2/2,places=6)


if __name__=='__main__':
    unittest.main()
