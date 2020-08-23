""" Verifying phase diagrams with plots from a freeware software """

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
from solvers.visuals import plot_mpltern

configs =[{'M': [1,1,10], 'chi':[0.2,0.3,1]}, {'M': [1,1,20], 'chi':[0.2,0.3,1]},
          {'M': [1,1,10], 'chi':[0.2,0.3,3.0]}, {'M': [1,1,64], 'chi':[0.2,0.3,1]}, 
         {'M': [40,400,4000], 'chi':[0.08,0.4,0.42]}]


""" some helper functions """
def compute_chemical_potential(phi,m,chi):
    mu1 = (phi[1]**2)*chi[0] + chi[1]*(phi[2]**2) + \
    phi[2]*(1-(1/m[2]) + phi[1]*(chi[0]+chi[1]-chi[2])) + np.log(phi[0])
    
    mu2 = chi[0]*(phi[1]-1)**2 + chi[1]*phi[2]**2 - phi[2]/m[2] + \
    phi[2]*((1 + (phi[1]-1))*(chi[0]+chi[1])+chi[2] - phi[1]*chi[2]) + np.log(phi[1])
    
    mu3 = 1 - phi[2] + m[2]*(-1 + chi[1] + chi[1]*phi[2]**2) + \
    m[2]*(phi[2]*(1-2*chi[1]+phi[1]*(chi[0] + chi[1]-chi[2])) + phi[1]*(chi[0]*(phi[1]-1)-chi[1] + chi[2])) + np.log(phi[2])
    
    return np.array([mu1,mu2,mu3])


import pprint
for i, c in enumerate(configs):
    configuration = c
    dimensions = len(configuration['M'])
    dx = 200
    pprint.pprint(configuration)
    output, simplices, grid, num_comps = compute(3, configuration, dx, True, thresh=10, flag_lift_label=False)

    ax, cbar = plot_mpltern(grid, simplices, num_comps)
    title = r'$\chi: $'+ ','.join('{:.2f}'.format(k) for k in configuration['chi'] )
    title = title +  '\n' + r'$\nu: $'+','.join('{:d}'.format(k) for k in configuration['M'])
    ax.set_title(title,pad=30)
    fname = '../figures/06-23-2020-Verify/'+'params_{}'.format(i) + '.png'
    plt.savefig(fname,dpi=500, bbox_inches='tight')
    plt.close()

    """ Plot chemical potentials """
    simplex_vertices = np.unique(np.asarray(simplices))
    chempots = [compute_chemical_potential(grid[:,x],configuration['M'],configuration['chi']) for x in simplex_vertices]
    chempots = np.asarray(chempots)

    simplex_coords = grid[:,simplex_vertices]
    fig = plt.figure(figsize=(12,4))
    fig.subplots_adjust(wspace=0.5)
    for j in range(chempots.shape[1]):
        ax = fig.add_subplot(1,3,j+1, projection='ternary')
        pc = ax.scatter(simplex_coords[2,:], simplex_coords[0,:], simplex_coords[1,:], c=chempots[:,j])
        ax.set_tlabel('solvent')
        ax.set_llabel('polymer')
        ax.set_rlabel('small molecule')
        ax.taxis.set_label_position('tick1')
        ax.laxis.set_label_position('tick1')
        ax.raxis.set_label_position('tick1')
        cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
        colorbar = fig.colorbar(pc, cax=cax)

    colorbar.set_label('Chemical potenital', rotation=270, va='baseline')
    fname = '../figures/06-23-2020-Verify/'+'chempots_{}'.format(i) + '.png'
    plt.savefig(fname,dpi=500, bbox_inches='tight')
    plt.close()


