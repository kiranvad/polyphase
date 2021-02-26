""" Bunch of visualization tools that aid analysis """
import matplotlib.pyplot as plt

import pdb
import numpy as np
import mpltern
from matplotlib.cm import ScalarMappable
from matplotlib import colors

from .helpers import *
from ._phase import is_boundary_point
     
def _set_axislabels_mpltern(ax):
    """ 
    Sets axis labels for phase plots using mpltern 
    in the order of solvent (index 2), polymer (index 0), non-solvent (index 1)
    """
    ax.set_tlabel(r'$\phi_2$', fontsize=15)
    ax.set_llabel(r'$\phi_1$', fontsize=15)
    ax.set_rlabel(r'$\phi_3$', fontsize=15)
    ax.taxis.set_label_position('tick1')
    ax.laxis.set_label_position('tick1')
    ax.raxis.set_label_position('tick1')

def plot_mpltern(grid, simplices, num_comps, ax = None):
    """ A phase diagram with simplices glued together with phase colorcoded 
    
    parameters:
    -----------
        grid          :  polyphase.PHASE.grid
        simplices     :  polyphase.PHASE.simplices
        num_comps     :  polyphase.PHASE.num_comps
        
    options:
    --------
        ax            :  matplotlib.pyplot.Axis object
    
    """

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection':'ternary'})
    else:
        fig = plt.gcf()
    
    phase_colors =['w','tab:red','tab:olive','tab:cyan']
    cmap = colors.ListedColormap(phase_colors[1:])
    triangle = np.array([[0, 0, 1], [1, 0, 0], [0,1,0]])
    #ax.fill(triangle[:,2], triangle[:,0], triangle[:,1], facecolor=phase_colors[0], edgecolor='none', alpha=0.75)
    for l,s in zip(num_comps, simplices):
        simplex_points = np.asarray([grid[:,x] for x in s])
        ax.fill(simplex_points[:,2], simplex_points[:,0], simplex_points[:,1], facecolor=phase_colors[int(l)])
    _set_axislabels_mpltern(ax)
    boundaries = np.linspace(1,4,4)
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
    cbar = fig.colorbar(mappable,shrink=0.5, aspect=5, ticks=[1.5,2.5,3.5],cax=cax)
    cbar.ax.set_yticklabels(['1-Phase', '2-Phase', '3-Phase'])
    
    return ax, cbar    
    

def plot_lifted_label_ternary(output, ax = None):
    """ A point cloud phase diagram from the lifted simplices 
    
    Input should the polyphase.PHASE.df instance
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection':'ternary'})
    else:
        fig = plt.gcf()
    
    #phase_colors =['tab:red','tab:olive','tab:cyan']
    phase_colors =['w','r','g','b']
    cmap = colors.ListedColormap(phase_colors[1:])
    df = output.transpose()
    for i, p in df.groupby('label'):
        ax.scatter(p['Phi_3'], p['Phi_1'], p['Phi_2'], c=phase_colors[int(i)])
    _set_axislabels_mpltern(ax)
    
    boundaries = np.linspace(1,4,4)
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable,shrink=0.5, aspect=5, ticks=[1.5,2.5,3.5],ax=ax)
    cbar.ax.set_yticklabels(['1-Phase', '2-Phase', '3-Phase'])
    
    return ax, cbar 

def plot_energy_landscape(outdict,mode='full', ax = None):
    """ Plots a convex hull of a energy landscape 
    
    parameters:
    -----------
        outdict     :  polyphase.PHASE.as_dict()
    
    This function takes an optional argument in mode which can be used to 
    visualize the just the convex hull (mode='convex_hull') approximation instead
    By default it plots the triangulated energy landscape (mode='full')
    This function plots the energy landscape with a thin boundary 
    cut around the two phase composotions
    """
    grid = outdict['grid']
    assert grid.shape[0]==3, 'Expected a ternary system but got {}'.format(grid.shape[0])

    boundary_points= np.asarray([is_boundary_point(x) for x in grid.T])
    energy = outdict['energy']
 
    if ax is None:
        fig = plt.figure(figsize=(4*1.6, 4))
        ax = fig.add_subplot(projection='3d')
    else:
        fig = plt.gcf()
        
    if mode=='full':    
        ax.plot_trisurf(grid[0,~boundary_points], grid[1,~boundary_points], 
                        energy[~boundary_points], linewidth=0.01, antialiased=True)
    elif mode=='convex_hull':
        ax.plot_trisurf(grid[0,:], grid[1,:], 
                        energy, triangles=outdict['simplices'], 
                        linewidth=0.01, antialiased=True)
    ax.set_xlabel('Polymer')
    ax.set_ylabel('Small molecule')
    ax.set_zlabel('Energy')
    ax.set_title('Energy landscape', pad=42)
    
    return ax, fig    
    
def plain_phase_diagram(df, ax = None):
    """ 
    Plot phase diagrams as points without any labels or stuff
    Used as a data point for dimensionality reduction and clustering
    
    parameters:
    -----------
        df     :  polyphase.PHASE.df (after calling .compute(), 
                  you should have access to the attribute .df if run with 'lift_label'=True)   

    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection':'ternary'})
    else:
        fig = plt.gcf()
    
    phase_colors =['w','r','g','b']
    for i, p in df.groupby('label'):
        ax.scatter(p['Phi_3'], p['Phi_1'], p['Phi_2'], c=phase_colors[int(i)])
        
    plt.axis('off')
    
    return ax     
    
    
    