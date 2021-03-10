import autograd.numpy as anp
from autograd import hessian
import polyphase
import matplotlib.pyplot as plt
from numpy.linalg import eig
from math import pi

class LSA:
    """Linear Stability Analysis of Flory-Huggins free enrgy
    Inputs:
    =======
        M   :  Degree of polymersization as a list of length N (= number of components)
        chi :  Flory-Huggins interaction parameters as a list one item per each combination of materials (Nc2 length)
        f   :  If you would like a different formulation of Flory-Huggins(this will be removed in due course)
        
    Methods:
    ========
        get_amplification_factor  :  Computes the amplication factor given a composition as a list
        evaluate : Evaluates the LSA and stores eigen values
        is_stable : Returns a boolean value whether a given composition is stable under LSA
        plot : Plots the eigen spectrum with in the wavelength values of [0,120]
        
    Attributes:
    ===========
        k : Range of wavelength vector used for LSA
        eps : Value of epsilon term in the Cahn-Hiliard formulation
        eigen_values : Eigen values matrix one entry for each k value
        
    Example:
    ========
        >>> import numpy as np
        >>> import polyphase
        >>> import pandas as pd
        
        # compute the phase diagram using CEM for reference
        >>> fh = lambda x: polyphase.flory_huggins(x,M,chi)
        >>> engine = polyphase.PHASE(fh,100,3)
        >>> engine.compute(use_parallel=False, verbose=False, lift_label=True)
        >>> df = engine.df.T
        
        # pick a point with in the composition and perform LSA
        >>> p1,p2,p3,l = df[df['label']==2].sample().values[0]
        >>> x0 = np.asarray([p1,p2,p3])
        >>> lsa = polyphase.LSA(M,chi,f=f)
        >>> lsa.evaluate(x0[:2])
        >>> lsa.plot()
        >>> print('Is the point stable: {}'.format(lsa.is_stable(x0[:2])))
    
    """
    def __init__(self, M,chi,f=None):
        self.M = M
        self.chi = chi
        if f is None:
            f = lambda x: self._flory_huggins(x)
        
        self.H = hessian(f)
        self.k = anp.linspace(0,120, num=50)
        self.eps = 1e-5
    
    def _flory_huggins(self,x):
        CHI = polyphase._utri2mat(self.chi, len(self.M))
        T1 = 0
        for xi, mi in zip(x,self.M):
            T1 += (xi*anp.log(xi))/mi
        T2 = 0.5*anp.matmul((anp.matmul(x,CHI)),anp.transpose(x)) 

        return T1+T2 
    
    def get_amplification_factor(self,x0,ki):
        hf = self.H(x0)
        A = (-(ki*pi)**2)*(hf + (self.eps**2)*((ki*pi)**2)*anp.identity(len(x0)))
        return A
    
    def evaluate(self,x0):
        """
        x0 : composiiton as a list of length N
        """
        eigen_values = []
        for ki in self.k:
            A = self.get_amplification_factor(x0,ki)
            w,v = eig(A)
            eigen_values.append(w)

        self.eigen_values = anp.asarray(eigen_values)
        
    def is_stable(self,x0):
        """
        x0 : composiiton as a list of length N
        """
        self.evaluate(x0)
        return ~(self.eigen_values[1:]>0).any()
    
    def plot(self):
        fig,ax = plt.subplots()
        for i in range(self.eigen_values.shape[1]):
            ax.plot(self.k, self.eigen_values[:,i], 
                    label=r'$\lambda_{}$'.format(i))
        ax.axhline(0,ls='--', lw=2.0, color='k')    
        ax.set_xlabel('k')
        ax.set_ylabel('eigen value')
        ax.legend(loc='best')
        plt.show()