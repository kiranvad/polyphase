import numpy as np
import pandas as pd
import pdb

from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
from pprint import pprint
import sys
if '../' not in sys.path:
    sys.path.insert(0,'../')

from solvers.visuals import plot_mpltern, _set_axislabels_mpltern

"""
1 :  Understanding simplicial topology of phases
"""
import cechmate as cm
from persim import plot_diagrams
rips = cm.Rips(maxdim=1, verbose=False) #Go up to 1D homology

def _show_simplex_mpltern(euclidean_coords,ax):
    ax.fill(euclidean_coords[:,2], euclidean_coords[:,0], euclidean_coords[:,1], facecolor='b', alpha=0.75)
    ax.scatter(euclidean_coords[:,2], euclidean_coords[:,0], euclidean_coords[:,1],c='k', s = 10, alpha=0.5)
    _set_axislabels_mpltern(ax)

def plot_rips_persistence(euclidean_coords,ax):
    rips.build(euclidean_coords)
    dgmsrips = rips.diagrams(show_inf=True)
    plot_diagrams(dgmsrips, ax=ax)
    ax.set_title("Rips Persistence Diagrams")
    
    return dgmsrips[0]

def get_H0curve(euclidean_coords):
    rips.build(euclidean_coords)
    dgmsrips = rips.diagrams()
    H0 = dgmsrips[0]
    cps = np.hstack((0.0,np.sort(H0[:,1]), 1.5))
    eps = 1e-5
    y = [np.sum(cps>x-eps) for x in cps]
    y[0] = 3
    y[-1] = 1
    
    x = cps.tolist()

    return x, y

def plot_simplicial_topolgy(vertices):
    fig = plt.figure(figsize=(3*3*1.6, 3))

    ax0 = fig.add_subplot(1,3,1, projection='ternary')
    _show_simplex_mpltern(vertices, ax0)
    
    ax1 = fig.add_subplot(1,3,2)
    #plot_simplicial_topology(simplex_euclidean_coords,ax1)
    x, y = get_H0curve(vertices)

    ax1.step(x,y)
    ax1.scatter(x,y)
    ax1.set_yticks([1, 2, 3])
    ax1.set_ylabel('Connected Components')
    ax1.set_xlabel('Distance threshold')
    
    ax2 = fig.add_subplot(1,3,3)
    H0 = plot_rips_persistence(vertices,ax2)
    
    return [ax0,ax1,ax2]
