import numpy as np
import pandas as pd
import polyphase
import unittest

import matplotlib.pyplot as plt

class TestVisuals(unittest.TestCase):
    def setUp(self):
        M = [5,5,1]
        chi = [1,0.5,0.5]
        f = lambda x: polyphase.flory_huggins(x, M, chi)
        self.engine = polyphase.PHASE(f, 50, 3)
        self.engine.compute()
        
    def test_plot_energy_landscape(self):
        polyphase.plot_energy_landscape(self.engine.as_dict(), mode='full')
        polyphase.plot_energy_landscape(self.engine.as_dict(), mode='convex_hull')
        
        print('function plot_energy_landscape passed')
        
    def test_plain_phase_diagram(self):
        polyphase.plain_phase_diagram(self.engine.df)
        print('function plain_phase_diagram passed')
        
    def test_TernaryPlot(self):
        ternplot = polyphase.TernaryPlot(self.engine)
        ternplot.plot_simplices()
        ternplot.plot_points()
        print('class TernaryPlot passed')
        
    def test_QuaternaryPlot(self):
        self.assertRaises(AssertionError, lambda : polyphase.QuaternaryPlot(self.engine))
        M = np.ones(4) 
        chi = 3.10*np.ones(int(0.5*4*(4-1)))
        f = lambda x: polyphase.flory_huggins(x, M, chi)
        engine = polyphase.PHASE(f, 50, 4)
        self.assertRaises(RuntimeError, lambda : polyphase.QuaternaryPlot(engine))
        engine.compute()
        qtplot = polyphase.QuaternaryPlot(engine)
        verts = qtplot.from4d23d([1,0,0,0])
        self.assertEqual(verts,[0,0,0])
        qtplot.plot_points()
        qtplot.plot_simplices()
        print('class QuaternaryPlot passed')
        
        
        
        
        
        