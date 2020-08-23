import pandas as pd
import numpy as np
import pickle
from numpy.linalg import norm
from scipy.constants import gas_constant    
import itertools
import pdb
from matplotlib import rc
rc('text', usetex=True)
import seaborn as sns
import matplotlib.pyplot as plt

import sys
if '../' not in sys.path:
    sys.path.insert(0,'../')

from solvers.cem import CEMLaplace
from solvers.helpers import flory_huggins
from solvers.utils import makegrid3d
    
    
# Load your data from dictonary to pandas data frame
with open('./data/htpdata/solubility.pkl', 'rb') as handle:
    data = pickle.load(handle)
    

def compute_chi(iSol,jSol,Vs = 100):
    """
    Inputs:
        iSol, jSol :  solubility parameters as a list (units MPa^{1/2})
        Vs :  Solvent molar volume in cm^3 / mol
    
    Units:
        iSol, jSol : both in Pa^{1/2} (Mega)
        thus delta_i,j will be in Pa^{1/2} (Mega)
        Formula for X_ij  = 0.34 + V_solvent/(RT) * (delta_i - delta_j)^2
                                   (m^3/mol)/((J/mol*K )* K) * (MPa^{1/2})^2
    """
    delta_i = norm(iSol,ord=None)
    delta_j = norm(jSol,ord=None)
    
    constant = 4.184*(2.045**2)/(8.314)
    
    chi_ij = 0.34 + (constant)*(Vs/(gas_constant*300)*(delta_i - delta_j)**2)
        
    return chi_ij

def plot_phase_diagram(spacing,chi,fname):
    grid = np.asarray(makegrid3d(num=spacing))
    M = [5,100,1] # M_p=100 , M_sm=5, M_s=1
    CHI = np.array([[0,chi[0],chi[1]],[chi[0],0,chi[2]],[chi[1],chi[2],0]])
    # chi is ordered as follows : chi_{s,sm}, chi_{s,p}, chi_{sm,p}
    labels = [r'$\varphi_{sm}$',r'$\varphi_{p}$',r'$\varphi_{s}$']

    gmix = lambda x: flory_huggins(x, M, CHI,beta=1e-4)
    energy = []
    for point in grid:
        energy.append(gmix(point))

    cem = CEMLaplace(grid,gmix,energy=energy)
    fig, ax = plt.subplots()
    ax = cem.plot_binodal(ax,fig,labels=labels)
    plt.axis('off')
    title = r'$\chi: $'+ ','.join('{:.2f}'.format(k) for k in chi )
    ax.set_title(title,pad=20)
    
    plt.savefig('./hteplots/'+fname+'.png',dpi=500,bbox_inches='tight')
    plt.close()

axes = [np.arange(0,len(data['solvents'])),np.arange(0,len(data['small molecules'])),\
        np.arange(0,len(data['polymers']))]
counter =1
for point in itertools.product(*axes):
    s = data['solvents'].loc[point[0]].tolist()
    sm = data['small molecules'].loc[point[1]].tolist()[2:5]
    p = data['polymers'].loc[point[2]].tolist()[2:5]
    fname = '{}_{}_{}'.format(point[0],data['small molecules']['name'].loc[point[1]],\
                            data['polymers']['name'].loc[point[2]])
    
    chi = []
    for comb in itertools.combinations([s,sm,p],2):
        chi.append(compute_chi(comb[0],comb[1]))
    print('Computing {} : {}'.format(counter,fname))
    plot_phase_diagram(100,chi,fname)
    counter += 1
    
    
    