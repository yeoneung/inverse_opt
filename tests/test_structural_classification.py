"""Edge cases and independent convex-program checks for the structural statements."""
import unittest
import numpy as np
from experiments.classification import enclosing,strip_risks


class StructuralClassificationTests(unittest.TestCase):
    def test_facet_obliqueness_and_degenerate_no_gain_directions(self):
        for beta in [0.,.1,1/3,.8,1.]:
            left=np.array([-np.sqrt(beta),0.]);right=np.array([np.sqrt(beta),np.sqrt(1-beta)])
            origin=np.zeros(2);points=np.array([left,origin,right])
            mixed,_=enclosing(points)
            proper=min(enclosing(points,[left,origin])[0],enclosing(points,[origin,right])[0])
            self.assertAlmostEqual(mixed,(1+3*beta)/4,places=7)
            self.assertAlmostEqual(proper,(1+3*beta)**2/(4*(1+beta)**2),places=7)
            if beta in [0.,1.]:self.assertAlmostEqual(proper,mixed,places=7)
            else:self.assertGreater(proper,mixed)

    def test_full_dimensional_tolerance_reversal_and_its_disappearance(self):
        upper=1.2;before_r,before_p,_=strip_risks(upper,None)
        for eta in [0.,.05,.19]:
            mixed,proper,_=strip_risks(upper,eta)
            self.assertAlmostEqual(mixed,(1+eta)**2/2,places=7)
            self.assertAlmostEqual(proper,1+eta+eta**2/2,places=7)
            self.assertLess(mixed,before_r);self.assertGreater(proper,before_p)
        for eta in [upper,2*upper]:
            mixed,proper,_=strip_risks(upper,eta)
            self.assertAlmostEqual(mixed,upper**2/2,places=7)
            self.assertAlmostEqual(proper,mixed,places=7)

    def test_class_reduction_removes_the_realizing_midpoint(self):
        beta=1/3;d=np.array([np.sqrt(beta),np.sqrt(1-beta)])
        left=np.array([-np.sqrt(beta),0.]);mid=(left+d)/2
        boundary=np.array([-np.sqrt(beta)/3,0.]);zero=np.zeros(2)
        contacts=np.array([left,zero,d,mid,boundary])
        before=min(enclosing(contacts,[left,zero])[0],enclosing(contacts,[d,mid,boundary,zero])[0])
        after=min(enclosing([left,zero,d],[left,zero])[0],enclosing([left,zero,d],[zero,d])[0])
        self.assertAlmostEqual(before,.5,places=7);self.assertAlmostEqual(after,9/16,places=7)
        B=np.eye(2)-np.outer(d,d)
        self.assertLess(np.linalg.norm(B@d),1e-12)
        self.assertGreater(np.linalg.norm(B@mid),.1)


if __name__=='__main__':unittest.main()
