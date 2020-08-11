""" Computes phase diagrams for three component system from Penn state """

import numpy as np
import pandas as pd
import pdb

from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt

import sys
if '../' not in sys.path:
    sys.path.insert(0,'../')
    
from parallel.parphase import compute
from solvers.visuals import plot_3d_phasediagram

from numpy.linalg import norm
from scipy.constants import gas_constant

def compute_chi(delta_i,delta_j,V):
    """
    total solubility parameters delta_i, delta_j are computed from hydrogen, polar, dispersive components
    
    delta_i, delta_j in MPa^{1/2} and V in cm3/mol
    
    returns a scalar chi value
    
    """
    constant = 1.0 #4.184*(2.045**2)/(8.314)
    
    chi_ij =  0.34+(constant)*(V/(gas_constant*300)*(delta_i - delta_j)**2)
        
    return chi_ij

def compute_weighted_chi(vec1,vec2,V, W):
    value = 0.0
    for i,w  in enumerate(W):
        value += w*(vec1[i]-vec2[i])**2
    
    value = 0.34 + value*(V/(gas_constant*300))
    
    return value
                   
from itertools import combinations
def get_chi_vector(deltas, V0, approach=1):
    """
    Given a list of deltas, computes binary interactions of chis
    """
    combs = combinations(deltas,2)
    inds = list((i,j) for ((i,_),(j,_)) in combinations(enumerate(deltas), 2))
      
    if approach==1:
        chi = [compute_chi(i[0],i[1],V0) for i in combs]
    elif approach==2:
        chi = [compute_weighted_chi(i[0],i[1],V0, W = [1.0,1.0,1.0]) for i in combs] 
    elif approach==3:
        chi = [compute_weighted_chi(i[0],i[1],V0, W = [1.0,0.25,0.25]) for i in combs]            
    else:
        raise KeyError
        
    return chi, inds

data_array = np.array([['Solvent', 'd', 'p', 'hb', 'T', 'VM'],
['Chlorobenzene', 19.0, 4.3, 2.0 ,19.58 ,81.48],
['o-Dichlorobenzene', 19.2, 6.5 ,3.3 ,20.47 ,92.86],
['Chloroform', 17.8, 3.1 ,5.7, 18.95, 63.05],
['o-Xylene', 18.0, 1.4, 2.9, 18.10, 94.00],
['Toluene', 18.0 ,1.4 ,2.0, 18.29, 81.42]])

df = pd.DataFrame(data=data_array[1:,1:],index=data_array[1:,0],columns=data_array[0,1:])
df['delta'] = df.apply(lambda x: norm([x['d'],x['p'],x['hb']]), axis=1)

""" Collect material properties """
import pprint
from solvers.visuals import plot_mpltern
delta_polymer = 19.34 #(MPa^1/2)
delta_sm = 19.64  #(MPa^1/2)
M = [946,12,1] #(polymer, small molecule, solvent, co-solvent)
delta_vec_polymer = [18.37,4.36,4.19]
delta_vec_sm = [19.98,3.72,3.44]

for solvent in df.index:
    V0 = float(df.loc[solvent,'VM'])
    for approach in range(1,4):
        if approach==1:
            delta_solvent = df.loc[solvent,'delta']
            chi = get_chi_vector([delta_polymer, delta_sm,delta_solvent],V0)[0]
        else:
            delta_solvent = df.loc[solvent,['d','p','hb']].astype('float').to_list()
            chi = get_chi_vector([delta_vec_polymer,delta_vec_sm,delta_solvent], V0, approach)[0]
    
        """ configure your material system """
        dimensions = len(M)
        configuration = {'M': M, 'chi':chi}
        pprint.pprint(configuration)
        dx = 400
        kwargs = {'flag_lift_label': False, 'use_weighted_delaunay': False,\
          'flag_remove_collinear' : False, 'beta':1e-4, \
          'flag_make_energy_paraboloid': True, 'flag_lift_purecomp_energy': False,\
                  'thresh' : 1}
        out = compute(3, configuration, dx, **kwargs)
        grid = out['grid']
        num_comps = out['num_comps']
        simplices = out['simplices']
        output = out['output']

        """ Post-processing """
        ax, cbar = plot_mpltern(grid, simplices, num_comps)
        ax.set_ternary_lim(
            0.8, 1.0,  # tmin, tmax
            0.0, 0.2,  # lmin, lmax
            0.0, 0.2,  # rmin, rmax
        )
        title = 'Approach : {} \n'.format(approach) + solvent + r'$\chi: $'+ ','.join('{:.2f}'.format(k) for k in chi )
        ax.set_title(title,pad=30)
        fname = '../figures/PM6Y6_v2/mpltern_'+ 'approach_{}'.format(approach) + solvent + '.png'
        plt.savefig(fname,dpi=500, bbox_inches='tight')
        plt.close()

        ax, cbar = plot_mpltern(grid, simplices, num_comps)

        title = solvent + r'$\chi: $'+ ','.join('{:.2f}'.format(k) for k in chi )
        ax.set_title(title,pad=30)
        fname = '../figures/PM6Y6_v2/'+ 'approach_{}'.format(approach)+solvent + '.png'
        plt.savefig(fname,dpi=500, bbox_inches='tight')
        plt.close()

print("Program ended!")