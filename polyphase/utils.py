import numpy as np
import pdb


def makegrid2d(num=50):
    X = np.linspace(0.001,0.999,num=num)
    Y = X
    grid = []
    for x in X:
        for y in Y:
            if np.isclose(x+y,1.0,atol=1e-3,rtol=1e-3):
                grid.append([x,y])
    return grid

def makegrid3d(num=50):
    X = np.linspace(0.001,0.999,num=num)
    Y,Z = X,X
    grid = []
    for x in X:
        for y in Y:
            for z in Z:
                if np.isclose(x+y+z,1.0,atol=1e-3,rtol=1e-3):
                    grid.append([x,y,z])
           
    return grid

""" Plot tools """
import mpltern
from matplotlib import rc
rc('text', usetex=True)
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_ternlabel(ax):
    ax.set_tlabel("$\\varphi_{p1}$",fontsize=15)
    ax.set_llabel("$\\varphi_{s}$",fontsize=15)
    ax.set_rlabel("$\\varphi_{p2}$",fontsize=15)
    ax.taxis.set_label_position('corner')
    ax.laxis.set_label_position('corner')
    ax.raxis.set_label_position('corner')
    sns.axes_style("ticks")
    
    return ax
    
def plotlocus(points,curvature,ax = None):
    t,l,r = points[:,0],points[:,1],points[:,2]
    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
    labels = ["+ve",'-ve','mixed','other']
    #colors = sns.mpl_palette("Set2",4)
    colors = sns.xkcd_palette(colors)
    for label in range(4):
        ids = np.where(np.asarray(curvature)==label)
        ax.scatter(t[ids[0]], l[ids[0]], r[ids[0]],c=colors[label],label=labels[label])
    ax = set_ternlabel(ax)
    return ax

def contours(points,colorcode, mode='lines', ax = None,level=None):
    t, l, r, v = points[:,0],points[:,1],points[:,2], colorcode
    if ax is None:
        fig = plt.figure(figsize=(10.8, 4.8))
        fig.subplots_adjust(left=0.075, right=0.85, wspace=0.3)
        ax = fig.add_subplot(projection='ternary')

    pad_title = 36

    if mode is 'lines':
        if level is not None:
            cs = ax.tricontour(t, l, r, v,levels=level)
        else:
            cs = ax.tricontour(t, l, r, v)
        ax.clabel(cs)
        ax.set_title("Contour lines", pad=pad_title)

        cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)

    elif mode is 'contour':
        cs = ax.tricontourf(t, l, r, v)
        cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
    else:
        raise KeyError("No such mode detected.")
    
    ax = set_ternlabel(ax)
    
    return ax,cs, cax

import plotly.graph_objects as go

def scatter3d(data,energy):
    data = np.asarray(data)

    trace = go.Scatter3d(
        x=data[:,0],
        y=data[:,1],
        z=energy,
        mode='markers',
        marker=dict(size=5,color=[],colorscale='RdBu',  opacity=0.8 ))

    return trace

from .helpers import get_ternary_coords, flory_huggins 
from .parphase import _utri2mat

def plot_energy_landscape(outdict):
    """ Plots a convex hull of a energy landscape """
    grid = outdict['grid']
    tri_coords = np.array([get_ternary_coords(pt) for pt in grid.T])
    energy = outdict['energy']
    simplices = outdict['simplices']
    #simplices = outdict['hull'].simplices
    
    fig = plt.figure(figsize=(4*1.6, 4))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(grid[0,:], grid[1,:], energy, triangles=simplices, linewidth=0.2,  antialiased=True, color=(0,0,0,0), edgecolor='Gray')
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\phi_2$')
    ax.set_zlabel('Energy')
    
    plt.show()



# Intereactive energy manifolds and convex hulls
import plotly.figure_factory as ff
from scipy.spatial import Delaunay

def plot_triangulated_surface(u, v, x,y,z):
    points2D = np.vstack([u,v]).T
    tri = Delaunay(points2D)
    simplices = tri.simplices
    fig = ff.create_trisurf(x=x, y=y, z=z,
                         simplices=simplices)
    
    return fig

def make_torus():
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, 2*np.pi, 20)
    u,v = np.meshgrid(u,v)
    u = u.flatten()
    v = v.flatten()

    x = (3 + (np.cos(v)))*np.cos(u)
    y = (3 + (np.cos(v)))*np.sin(u)
    z = np.sin(v)
    
    return u, v, x, y, z

def make_mobious_strip():
    u = np.linspace(0, 2*np.pi, 24)
    v = np.linspace(-1, 1, 8)
    u,v = np.meshgrid(u,v)
    u = u.flatten()
    v = v.flatten()

    tp = 1 + 0.5*v*np.cos(u/2.)
    x = tp*np.cos(u)
    y = tp*np.sin(u)
    z = 0.5*v*np.sin(u/2.)
    
    return u, v, x, y, z

def test_plot_triangulated_surface():
    u, v, x, y, z = make_mobious_strip()
    fig = plot_triangulated_surface(u, v, x, y, z)
    fig.update_layout(title=config_str,scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title = "z"),
        coloraxis_colorbar=dict(title='z'),
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple")
    )
    fig.write_html('../figures/3dplots/test_mobious.html')    
    
    
    
    
""" utilities for some example runs """

def get_data(name='ow',fhid=0):
    """
    For a given data name and produces a M and Chi parameters of a ternary system.
    If the name is  FHPaper, it requires an id in fhid (default 0 ) in the range of (0,4)
    Get a dummy sets of data for trials:

    from solvers.utils import get_data
    M,CHI=get_data(name='ow',fhid=0)
    
    OW APS Physics Letters: https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=1231&context=me_pubs
    B Zhou Page:65 
    https://dspace.mit.edu/bitstream/handle/1721.1/35312/76904537-MIT.pdf?sequence=2&isAllowed=y
    
    """
    floryhuggins =[{'name': 'CHCl3/F8/PCBM','chi': [0.341,0.885,0.941]},
                  {'name': 'chlorobenzene/PFB/PCBM','chi': [0.340,0.899,0.899]},
                  {'name': 'CHCl3/APFO-3/PCBM','chi': [0.505,0.885,0.450]},
                  {'name': 'chlorobenzene/APFO-3/PCBM','chi': [0.480,0.899,0.479]}, 
                  {'name': 'xylene/PFB/PCBM','chi': [0.442,1.630,1.013]}
                   ]
    N = {'CHCl3':1,'chlorobenzene':1,'xylene':1,'F8':720,'PFB':73,'F8BT':172,'APFO-3':66,'PCBM':5}
    fname = name
    if name is 'ow':
        M = np.array([5,1,5])
        chi = [0.5,1,0.5] # chi_12, chi_13, chi_23
    elif name is 'bz65':
        M = np.array([1,1,64])
        chi = [0.2,0.3,1] # chi_12, chi_13, chi_23
    elif name is 'FHPaper':
        chiset = floryhuggins[fhid]
        solvent,polymer,PCBM = chiset['name'].split("/")
        M = [N[solvent],N[polymer],N[PCBM]]
        #M = np.array([1.0,64,5])
        chi = np.asarray(chiset['chi'])
        fname = "_".join([solvent,polymer,PCBM])
    elif name is 'temp':
        M = np.array([1.0,64.0,1.0])
        chi = [1.0,0.2,0.3] # chi_12, chi_13, chi_23
    else:
        raise KeyError('No such data exists')
    CHI = np.array([[0,chi[0],chi[1]],[chi[0],0,chi[2]],[chi[1],chi[2],0]])    
    
    info = {'params':r'M:{},$\chi$:{}'.format(M,chi),'fname':fname}
    
    return M, CHI, info


def compute_chemical_potential(phi,m,chi):
    mu1 = (phi[1]**2)*chi[0] + chi[1]*(phi[2]**2) + \
    phi[2]*(1-(1/m[2]) + phi[1]*(chi[0]+chi[1]-chi[2])) + np.log(phi[0])
    
    mu2 = chi[0]*(phi[1]-1)**2 + chi[1]*phi[2]**2 - phi[2]/m[2] + \
    phi[2]*((1 + (phi[1]-1))*(chi[0]+chi[1])+chi[2] - phi[1]*chi[2]) + np.log(phi[1])
    
    mu3 = 1 - phi[2] + m[2]*(-1 + chi[1] + chi[1]*phi[2]**2) + \
    m[2]*(phi[2]*(1-2*chi[1]+phi[1]*(chi[0] + chi[1]-chi[2])) + phi[1]*(chi[0]*(phi[1]-1)-chi[1] + chi[2])) + np.log(phi[2])
    
    return np.array([mu1,mu2,mu3])