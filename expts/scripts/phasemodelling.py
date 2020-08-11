import autograd.numpy as anp
import numpy as np
import pdb
from autograd import elementwise_grad as egrad
from autograd import grad, jacobian, hessian

class twod_precomputed:
    def __init__(self,CHI,M):
        self.chi = CHI
        self.m = M
        
    def freeenergy(self,point):
        F = (1/self.m[0])*(point[0]*np.log(point[0]))
        F += (1/self.m[1])*(point[1]*np.log(point[1]))
        F += self.chi[0,1]*point[0]*point[1]
        
        return F
        
    def hessian(self,point):
        h11 = 1/(self.m[0]*point[0])
        h12 = self.chi[0,1]
        h22 = 1/(self.m[1]*point[1])

        H = [[h11,h12],[h12,h22]]

        return H

    def spinodal(self,point):
        value = self.chi[0,1] - 0.5*(1/(self.m[0]*point[0]) + 1/(self.m[1]*point[1]))
        if np.absolute(value)<1:
            flag = 0
        else:
            flag = 1
        
        return flag, value


class phase:
    def __init__(self,chi,m, func = None):
        self.chi = chi # Flory huggins parameter
        self.m = m # Degree of polymerization
        if func is None:
            self.func = lambda p: self.free_energy(p)
        else:
            self.func = func
        
    def free_energy(self,phi):
        """
        computes a free energy per mole and returns a scalar value.
        """
        T1 = anp.dot(anp.true_divide(phi, self.m),anp.log(phi))
        T2 = 0.5*np.matmul((anp.matmul(phi,self.chi)),anp.transpose(phi))

        output = T1+T2

        return output

    def spinodal(self,point):
        """
        Computes the spinodal equation criteria for ternary using a Hessian of Free energy
        """
        hes = hessian(self.func)
        d3GdP3 = jacobian(hes)
        pdb.set_trace()
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
            
        return curvature, H, eigs
    
    def critical(self,point,tol=1e-3):
        """
        Computes a criteria for critical point of ternary using a Jacobian and Hessian
        """
        J = jacobian(self.func)(point)
        tols = tol*np.ones(len(point))
        flag = np.allclose(np.absolute(J),tols)
  
        return flag, J

    def binodal(self,point):
        """
        compute binodal region of the phase
        """
        raise NotImplementedError()



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    