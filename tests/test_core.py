import numpy as np
import pandas as pd
import polyphase
import unittest
import pdb

def f(x):
    M = [5,5,1]
    chi = [1,0.5,0.5]
    x1,x2,x3 = x
    entropy = (x1*np.log(x1))/M[0] + (x2*np.log(x2))/M[1] + (x3*np.log(x3))/M[2]
    enthalpy = chi[0]*x1*x2 + chi[1]*x1*x3 + chi[2]*x2*x3
    
    return entropy+enthalpy

class TestCore(unittest.TestCase):
    def setUp(self):
        self.engine = polyphase.PHASE(f, 50, 3)
        
    def test_PHASE(self):
        self.assertEqual(self.engine.meshsize,50)
        self.assertEqual(self.engine.dimension,3)
        self.assertFalse(self.engine.is_solved)
        
    def test_compute_defaults(self):
        self.engine.compute()
        self.assertFalse(self.engine.use_parallel)
        self.assertFalse(self.engine.verbose)
        self.assertTrue(self.engine.lift_label)
        self.assertEqual(self.engine.thresh_scale, 0.1*50)
  
        self.assertTrue(self.engine.is_solved) 
        np.testing.assert_array_equal(np.unique(self.engine.num_comps), np.array([1,2]))
        self.assertEqual(np.sum(self.engine.coplanar),0)
        np.testing.assert_array_equal(self.engine.df.T['label'].unique(), np.array([0,1,2]))
        
    def test_get_phase_compositions(self):
        x = np.asarray([0.333,0.333,0.334])
        self.assertRaises(RuntimeError, lambda : self.engine.get_phase_compositions(x))
        self.engine.compute()
        x_boundary = np.asarray([0.5,0.5,0.0])
        self.assertRaises(RuntimeError, lambda : self.engine.get_phase_compositions(x_boundary))
        x_comps, vertices, num_comps = self.engine.get_phase_compositions(x)
        self.assertEqual(len(x),3)
        self.assertEqual(len(vertices),3)
        self.assertEqual(num_comps,2)
        
    def test_parallel(self):
        self.engine.compute(use_parallel=False, lower_hull_method='point_at_infinity') 
        serial = self.engine.as_dict()
        self.engine.compute(use_parallel=True)
        parallel = self.engine.as_dict()
        np.testing.assert_array_equal(serial['num_comps'], parallel['num_comps'])
        np.testing.assert_array_equal(serial['grid'], parallel['grid'])
        np.testing.assert_array_equal(serial['energy'], parallel['energy'])
        np.testing.assert_array_equal(serial['simplices'], parallel['simplices'])
        np.testing.assert_array_equal(serial['thresh'], parallel['thresh'])
        pd._testing.assert_frame_equal(serial['output'], parallel['output'])
        
if __name__ == '__main__':
    unittest.main()        