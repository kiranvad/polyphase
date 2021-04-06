import numpy as np
import pandas as pd
import polyphase
import unittest
import autograd.numpy as anp

M = [5,5,1]
chi = [1,0.5,0.5]

from autograd import jacobian
def f(x):
    x1,x2 = x
    x3 = anp.abs(1-x1-x2).astype(float)
    entropy = (x1*anp.log(x1))/M[0] + (x2*anp.log(x2))/M[1] + (x3*anp.log(x3))/M[2]
    enthalpy = chi[0]*x1*x2 + chi[1]*x1*x3 + chi[2]*x2*x3
    
    return entropy+enthalpy

class TestTests(unittest.TestCase):
    def setUp(self):
        self.fh = lambda x : polyphase.flory_huggins(x, M, chi)
        self.engine = polyphase.PHASE(self.fh, 100, 3)
        self.engine.compute()
        
    def test_CentralDifference(self):
        cd = polyphase.CentralDifference(self.fh)
        p1,p2,p3 = [0.45,0.45,0.1]
        [dfdx_1, dfdy_1] = cd([p1,p2,p3])
        [dfdx_2, dfdy_2] = jacobian(f)(np.asarray([p1,p2]))
        
        np.testing.assert_allclose(dfdx_1,dfdx_2,rtol=1e-4)
        np.testing.assert_allclose(dfdy_1,dfdy_2,rtol=1e-4)
        
        print('class CentralDifferences passed')
        
    def test_TestAngles(self):
        gradient = polyphase.CentralDifference(self.fh)
        TEST = polyphase.TestAngles(self.engine, simplex_id=0)
        assert TEST.rnd_simplex_indx==0, "Simplex index should be correctly set to 0"
        TEST = polyphase.TestAngles(self.engine, phase=1)
        assert TEST.rnd_simplex_indx<=len(self.engine.simplices), "Simplex index needs to be lesser than number of simplices"
        assert np.shape(TEST.vertices)[1]==self.engine.dimension, "Points must be of same dimension as the system"
        assert np.shape(np.asarray(TEST.parametric_points))[1]==self.engine.dimension, "Points must be of same dimension as the system"
        assert TEST.rnd_simplex in self.engine.simplices , "Selected simplex should be in the engine.simplices"
        TEST.base_visualize()
        normal,dx,dy = TEST._get_normal([0.45,0.45,0.1], gradient)
        angle = TEST._angle_between_vectors(normal,[1,0,dx])
        np.testing.assert_allclose(angle,90,rtol=1e-4)
        TEST.get_angles(gradient)
        TEST.visualize()
        
    def test_TestPhaseSplits(self):
        PHASE_ID = 2
        phase_simplex_ids = np.where(np.asarray(self.engine.num_comps)==PHASE_ID)[0]
        simplex_id = phase_simplex_ids[0]
        phasesplits = polyphase.TestPhaseSplits(self.engine,phase=PHASE_ID,
                                                simplex_id=simplex_id, threshold=0.05)
        phasesplits.check_centroid()
        x_obtained = phasesplits.centroid_splits_
        x_expected = [0.33,0.66,0.02]
        np.testing.assert_allclose(x_obtained, x_expected, rtol=1e-1, atol=0)
        phasesplits.visualize_centroid()
        
        
        
        
        
        
        
        
        
        
        