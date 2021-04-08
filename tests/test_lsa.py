import numpy as np
import polyphase
import unittest
import pandas as pd
import autograd.numpy as anp

M = [5,5,1]
chi = [1,0.5,0.5]

def f(x):
    x1,x2 = x
    x3 = anp.abs(1-x1-x2).astype(float)
    entropy = (x1*anp.log(x1))/M[0] + (x2*anp.log(x2))/M[1] + (x3*anp.log(x3))/M[2]
    enthalpy = chi[0]*x1*x2 + chi[1]*x1*x3 + chi[2]*x2*x3
    
    return entropy+enthalpy

class TestLSA(unittest.TestCase):
    def setUp(self):
        fh = lambda x : polyphase.flory_huggins(x, M, chi)
        self.engine = polyphase.PHASE(fh,100,3)
        self.engine.compute()
        df = self.engine.df.T
        p1,p2,p3,l = df[df['label']==2].sample().values[0]
        self.point = np.asarray([p1,p2])
        
    def test_lsa(self):
        lsa = polyphase.LSA(M,chi,f=f)
        A = lsa.get_amplification_factor(self.point,5)
        np.testing.assert_equal(A.shape[0],A.shape[1])
        lsa.evaluate(self.point)
        num_k, num_eigs = np.shape(lsa.eigen_values)
        self.assertEqual(num_k, 50)
        self.assertEqual(num_eigs, 2)
        self.assertFalse(lsa.is_stable(self.point))
        lsa.plot()
        
        print('class LSA passed')
        
        