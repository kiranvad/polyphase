import autograd.numpy as anp
import numpy as np
import pdb
from autograd import elementwise_grad as egrad
from autograd import grad, jacobian, hessian

class Spinodal:
    def __init__(self,grid, gmix, energy=None):
        """
        Implementation of Hessian based stability analysis for spinodal points computation.
        
        Inputs;
        --------
        grid   :  Grid of points where free enery needs to be computed. A nd array.
        gmix   :  Free enrgy functional formation with a n-dimensional point as input
        
        """
        self.grid = grid
        if energy is None:
            energy = []
            for point in self.grid:
                energy.append(gmix(point))
            self.energy = energy
            
        else:
            self.energy = energy
        
        self.func = gmix

    def _get_spinodal(self,point):
        """
        Computes the spinodal equation criteria for ternary using a Hessian of Free energy
        """
        hes = hessian(self.func)
        H = hes(point)
        
        eigs = anp.linalg.eigvals(H)

        if np.all(eigs>0):
            curvature = 0
        elif np.all(eigs<0):
            curvature = 1
        elif np.any(eigs>0):
            curvature = 2
        else:
            curvature = 3
        
        output = {'curvature':curvature,'Hessian':H,'eigs':eigs}
        
        return output
    
    def get_spinodal(self):
        score = []
        for point in self.grid:
            out = self._get_spinodal(point)
            score.append(out['curvature'])
            
        return score
        
        