import autograd.numpy as anp
from autograd import grad, hessian
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
    
    """
    def __init__(self, M,chi):
        self.M = M
        self.chi = chi
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
        return (anp.prod(self.eigen_values, axis=1)<=0).all()
    
    def plot(self):
        fig,ax = plt.subplots()
        for i in range(3):
            ax.plot(self.k, self.eigen_values[:,i], 
                    label=r'$\lambda_{}$'.format(i))
        ax.axhline(0,ls='--', lw=2.0, color='k')    
        ax.set_xlabel('k')
        ax.set_ylabel('eigen value')
        ax.legend(loc='best')
        plt.show()