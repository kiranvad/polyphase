""" Bunch of visualization tools that aid analysis """
import matplotlib.pyplot as plt

import pdb
import numpy as np
import mpltern
from matplotlib.cm import ScalarMappable
from matplotlib import colors

from .helpers import *

def plot_4d_phase_simplex_addition(pm,sliceat=0.5):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib import colors

    """ 
    Given a PhaseModelling object, plots the phase diagram by just gluing simplices together 
    """
    if not pm.dimension==4:
        raise NameError('Dimension mismatch : This function works only for 4-component systems')
        
    fig, axs = plt.subplots(2,2,subplot_kw={'projection': '3d'}, figsize=(8,8))
    fig.subplots_adjust(hspace=0.2,wspace=0.2)
    v = np.array([[0, 0, 0], [1, 0, 0], [1/2,np.sqrt(3)/2,0],  [1/2,np.sqrt(3)/6,np.sqrt(6)/3]])
    axs = axs.reshape(4)
    words = [r'$\varphi_{1}$',r'$\varphi_{2}$',r'$\varphi_{3}$',r'$\varphi_{4}$']

    for ax in axs:
        ax.scatter3D(v[:, 0], v[:, 1], v[:, 2],color='black')
        ax._axis3don = False
        #ax.view_init(elev=25, azim=75)
        verts = get_convex_faces(v)
        ax.add_collection3d(Poly3DCollection(verts, facecolors='black', linewidths=0.5, edgecolors='black', alpha=.05))
        for vi,w in zip(v,words):
            ax.text(vi[0],vi[1],vi[2],w,fontsize=20)
            
    phase_colors =['tab:red','tab:olive','tab:cyan','tab:purple']
    for i,simplex in zip(pm.num_comps,pm.simplices):
        vertices = [pm.grid[:,x] for x in simplex]
        v = np.asarray([from4d23d(vertex) for vertex in vertices])
        if np.all(np.asarray(vertices)[:,3]<sliceat):
            verts = get_convex_faces(v)
            axs[i-1].add_collection3d(Poly3DCollection(verts, facecolors=phase_colors[i-1], edgecolors=None))
    cmap = colors.ListedColormap(phase_colors)
    boundaries = np.linspace(1,5,5)
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable,shrink=0.5, aspect=5, ticks=[1.5,2.5,3.5,4.5],ax=[axs[1],axs[3]])
    cbar.ax.set_yticklabels(['1-Phase', '2-Phase', '3-Phase','4-Phase'])
  
    plt.show()

""" Ternary plots for 3-component system """
    
def plot_3d_phasediagram(grid, simplices, num_comps, ax = None):
    """ Depreciated fucntion """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    ax.set_aspect('equal')
    
    coords = np.asarray([get_ternary_coords(pt) for pt in grid.T])
    tpc = ax.tripcolor(coords[:,0], coords[:,1], simplices, facecolors=np.asarray(num_comps), edgecolors='none')
    cbar = fig.colorbar(tpc, ticks=[1, 2, 3])
    cbar.ax.set_yticklabels(['1-phase', '2-phase', '3-phase'])
    cbar.set_label('Phase region identification')

    words = [r'$\varphi_{1}$',r'$\varphi_{2}$',r'$\varphi_{3}$']
    xs = [-0.15,1,0.5]
    ys = [0,0,np.sqrt(3)/2+0.01]
    for x, y, s in zip(xs,ys,words):
        ax.text(x,y,s,fontsize=20)

    plt.axis('off')
    
    return ax, cbar    
    
def _set_axislabels_mpltern(ax):
    """ 
    Sets axis labels for phase plots using mpltern 
    in the order of solvent (index 2), polymer (index 0), non-solvent (index 1)
    """
    ax.set_tlabel('solvent', fontsize=20)
    ax.set_llabel('polymer', fontsize=20)
    ax.set_rlabel('small molecule', fontsize=20)
    ax.taxis.set_label_position('tick1')
    ax.laxis.set_label_position('tick1')
    ax.raxis.set_label_position('tick1')

def plot_mpltern(grid, simplices, num_comps, ax = None):
    
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection':'ternary'})
    else:
        fig = plt.gcf()
    
    """ A phase diagram with simplices glued together with phase colorcoded """
    phase_colors =['tab:red','tab:olive','tab:cyan']
    cmap = colors.ListedColormap(phase_colors)
    triangle = np.array([[0, 0, 1], [1, 0, 0], [0,1,0]])
    ax.fill(triangle[:,2], triangle[:,0], triangle[:,1], facecolor=phase_colors[0], edgecolor='none', alpha=0.75)
    for l,s in zip(num_comps, simplices):
        simplex_points = np.asarray([grid[:,x] for x in s])
        ax.fill(simplex_points[:,2], simplex_points[:,0], simplex_points[:,1], facecolor=phase_colors[l-1])
    _set_axislabels_mpltern(ax)
    boundaries = np.linspace(1,4,4)
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable,shrink=0.5, aspect=5, ticks=[1.5,2.5,3.5],ax=ax)
    cbar.ax.set_yticklabels(['1-Phase', '2-Phase', '3-Phase'])
    
    return ax, cbar    
    

def plot_lifted_label_ternary(output, ax = None):
    """ A point cloud phase diagram from the lifted simplices 
    
    Input should the output attribute from the compute function
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection':'ternary'})
    else:
        fig = plt.gcf()
    
    #phase_colors =['tab:red','tab:olive','tab:cyan']
    phase_colors =['r','g','b']
    cmap = colors.ListedColormap(phase_colors)
    df = output.transpose()
    for i, p in df.groupby('label'):
        ax.scatter(p['Phi_3'], p['Phi_1'], p['Phi_2'], c=phase_colors[int(i-1)])
    _set_axislabels_mpltern(ax)
    
    boundaries = np.linspace(1,4,4)
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable,shrink=0.5, aspect=5, ticks=[1.5,2.5,3.5],ax=ax)
    cbar.ax.set_yticklabels(['1-Phase', '2-Phase', '3-Phase'])
    
    return ax, cbar 

    
    
    
    
    
    