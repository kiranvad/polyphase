import numpy as np
import unittest
import polyphase

class TestUtils(unittest.TestCase):
    def test_utri2mat(self):
        ret = polyphase._utri2mat([1,1,1],3)
        np.testing.assert_array_equal(ret,np.array([[0,1,1],[1,0,1],[1,1,0]]))
        print('function _utri2mat passed')
        
    def test_flory_huggins(self):
        M = [5,5,1]
        chi = [1,0.5,0.5]
        x = [0.45,0.45,0.1]
        x1,x2,x3 = x
        entropy = (x1*np.log(x1))/M[0] + (x2*np.log(x2))/M[1] + (x3*np.log(x3))/M[2]
        enthalpy = chi[0]*x1*x2 + chi[1]*x1*x3 + chi[2]*x2*x3
        
        energy_1 = entropy + enthalpy
        energy_2 = polyphase.flory_huggins(x, M,chi,beta=0.0, logapprox=False)
        
        self.assertEqual(energy_1, energy_2)
        
        print('function polyphase.flory_huggins passed')
        
    def test_get_chi_vector(self):
        deltas = [[1,1,1],[1,1,1],[1,1,1]]
        chi_1,_ = polyphase.get_chi_vector(deltas, 1, approach=1)
        chi_2,_ = polyphase.get_chi_vector(deltas, 1, approach=2)
        chi_3,inds = polyphase.get_chi_vector(deltas, 1, approach=3)
        
        self.assertEqual(chi_1,[0.34,0.34,0.34])
        self.assertEqual(chi_2,[0.34,0.34,0.34])
        self.assertEqual(chi_3,[0.34,0.34,0.34])
        self.assertEqual(inds,[(0,1),(0,2),(1,2)])
        
        print('function polyphase.get_chi_vector passed' )
        