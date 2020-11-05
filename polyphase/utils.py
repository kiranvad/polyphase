import numpy as np
import pdb
from itertools import combinations

""" Plot tools """
import mpltern
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.spatial import Delaunay

from .helpers import get_ternary_coords 
from .parphase import _utri2mat
from scipy.constants import gas_constant

def set_ternlabel(ax):
    ax.set_tlabel("$\\varphi_{p1}$",fontsize=15)
    ax.set_llabel("$\\varphi_{s}$",fontsize=15)
    ax.set_rlabel("$\\varphi_{p2}$",fontsize=15)
    ax.taxis.set_label_position('corner')
    ax.laxis.set_label_position('corner')
    ax.raxis.set_label_position('corner')
    sns.axes_style("ticks")
    
    return ax

def plot_triangulated_surface(u, v, x,y,z, **kwargs):
    points2D = np.vstack([u,v]).T
    tri = Delaunay(points2D)
    simplices = tri.simplices
    fig = ff.create_trisurf(x=x, y=y, z=z,
                         simplices=simplices, **kwargs)
    
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
    

    
""" Compute chi from solubilities """

def _compute_chi(delta_i,delta_j,V):
    """
    total solubility parameters delta_i, delta_j are computed from hydrogen, polar, dispersive components
    
    delta_i, delta_j in MPa^{1/2} and V in cm3/mol
    
    returns a scalar chi value
    
    """
    constant = 1.0 #4.184*(2.045**2)/(8.314)
    chi_ij =  0.34+(constant)*(V/(gas_constant*300)*( np.asarray(delta_i) - np.asarray(delta_j) )**2)
        
    return chi_ij

def _compute_weighted_chi(vec1,vec2,V, W):
    value = 0.0
    for i,w  in enumerate(W):
        value += w*(vec1[i]-vec2[i])**2
    
    value = 0.34 + value*(V/(gas_constant*300))
    
    return value
                   
def get_chi_vector(deltas, V0, approach=1):
    """
    Given a list of deltas, computes binary interactions of chis
    """
    combs = combinations(deltas,2)
    inds = list((i,j) for ((i,_),(j,_)) in combinations(enumerate(deltas), 2))
      
    if approach==1:
        chi = [_compute_chi(np.linalg.norm(i[0]),np.linalg.norm(i[1]),V0) for i in combs]
    elif approach==2:
        chi = [_compute_weighted_chi(i[0],i[1],V0, W = [1.0,1.0,1.0]) for i in combs] 
    elif approach==3:
        chi = [_compute_weighted_chi(i[0],i[1],V0, W = [1.0,0.25,0.25]) for i in combs]            
    else:
        raise KeyError
        
    return chi, inds
    

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