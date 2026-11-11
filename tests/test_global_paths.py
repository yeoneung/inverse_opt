"""Independent checks of prescribed paths and global recovery bounds."""
import unittest
from fractions import Fraction as Q
import numpy as np
from src.projection_paths import (exact_chain_check,float_lift,proper_radius,
    mixed_radius,qp_check,polyhedral_line_path)
from experiments.capacity import capacity_instance,sample_forward


class GlobalPathTests(unittest.TestCase):
    def test_rational_normal_conditions_for_repeated_and_multidimensional_chains(self):
        rng=np.random.default_rng(20261110)
        for q in [1,2,4,7]:
            for m in [2,4,7]:
                result=exact_chain_check(rng.integers(-5,6,size=(m+1,q)).tolist(),Q(1,5*m*m))
                self.assertEqual(result['maximum_normal_violation'],'0')
        self.assertEqual(exact_chain_check([[0,0],[1,0],[0,0],[-1,0],[0,0]],Q(1,100))['maximum_normal_violation'],'0')

    def test_lifted_circle_matches_full_hull_qps_and_global_rate_bounds(self):
        for m in [4,8,16]:
            angle=np.arange(m+1)*2*np.pi/m
            points=np.c_[np.cos(angle),np.sin(angle)];points[-1]=points[0]
            vertices,M,intervals=float_lift(points,m**-3)
            P,_=proper_radius(vertices);Rlo,Rhi,center=mixed_radius(vertices)
            base=1+3*np.cos(np.pi/m)**2;extent=(m*m+m**4)/m**6
            self.assertAlmostEqual(proper_radius(points)[0],base,places=10)
            self.assertLessEqual(Rhi,1+extent/4+1e-8)
            self.assertGreaterEqual(P/Rhi,base/(1+extent/4)-1e-8)
            self.assertLess(P/Rlo,(1+np.sqrt(1-1/(2*m*m)))**2+1e-7)
            error,violation=qp_check(vertices,M,intervals,[0,m//2,m-1])
            self.assertLess(error,1e-7);self.assertLessEqual(violation,1e-12)

    def test_inferred_regions_match_independent_forward_queries(self):
        for q,seed,scale in [(4,771100,.6),(16,771105,1.6),(32,771107,.6)]:
            A,b,c,d,left,right=capacity_instance(q,seed,scale)
            vertices,queries,kkt,gap,pieces=polyhedral_line_path(A,b,c,d,left,right,return_pieces=True)
            times=np.linspace(left,right,17);actual=sample_forward(A,b,c,d,times)
            for t,x in zip(times,actual):
                valid=[p for p in pieces if p[0]-1e-8<=t<=p[1]+1e-8]
                self.assertTrue(valid)
                self.assertLess(min(np.linalg.norm(p[2]+t*p[3]-x) for p in valid),2e-6)
            self.assertLess(kkt,1e-10);self.assertLess(gap,1e-7)

    def test_contact_mean_is_feasible_and_degenerate_paths_are_supported(self):
        vertices=np.array([[0.,0.],[1.,0.],[0.,1.],[.2,.1]])
        lower,upper,mean=mixed_radius(vertices)
        self.assertTrue((mean>=0).all());self.assertLessEqual(mean.sum(),1+1e-12)
        self.assertLessEqual(lower,upper+1e-12)
        self.assertEqual(proper_radius(np.array([[.1,.2],[.1,.2]]))[0],0.)
        self.assertEqual(mixed_radius(np.array([[.1,.2]]))[0],0.)


if __name__=='__main__':unittest.main()
