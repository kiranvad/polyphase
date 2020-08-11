import ray
ray.init(ignore_reinit_error=True)

import mpltern
from matplotlib import rc
rc('text', usetex=True)
import seaborn as sns
import matplotlib.pyplot as plt

import pdb
import numpy as np

import sys
if '../' not in sys.path:
    sys.path.append('../')

from solvers.utils import makegrid3d, get_data

import warnings
warnings.filterwarnings("ignore")


from scipy.spatial import ConvexHull
import numpy as np
from math import pi

def get_ternary_coords(point):
    a,b,c = point
    x = 0.5-a*np.cos(pi/3)+b/2;
    y = 0.866-a*np.sin(pi/3)-b*(1/np.tan(pi/6)/2);
    
    return [x,y]

def flory_huggins(x, M,CHI,beta=1e-3):
    T1 = 0
    for i,xi in enumerate(x):
        T1 += (xi*np.log(xi))/M[i] + beta/xi
    T2 = 0.5*np.matmul((np.matmul(x,CHI)),np.transpose(x)) 
    
    return T1+T2  

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

@ray.remote
def setup_data(M,CHI,meshsize=40):
    grid = np.asarray(makegrid3d(num=meshsize))
    gmix = lambda x: flory_huggins(x, M, CHI,beta=1e-4)
    energy, coords = [],[]
    for point in grid:
        energy.append(gmix(point))
        coords.append(get_ternary_coords(point))

    points = np.concatenate((coords,np.asarray(energy).reshape(-1,1)),axis=1)
    hull = ConvexHull(points)
    coords = np.asarray(coords)

    return grid, energy, hull,coords

@ray.remote
def refine_simplices(hull,grid):
    simplices = []
    for simplex in hull.simplices:
        point_class = np.sum(np.isclose(grid[simplex],0.001),axis=1)
        point_class = np.unique(3 - point_class)
        if (point_class==1).any():
            pass
        else:
            simplices.append(simplex)
    return simplices

from scipy.spatial.distance import pdist, euclidean, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@ray.remote
def label_simplex(simplex,coords):
    thresh = 5*euclidean(coords[0,:],coords[1,:])
    tri_coords = coords[simplex,:]
    dist = squareform(pdist(tri_coords,'euclidean'))
    adjacency = dist<thresh
    adjacency =  adjacency.astype(int)  
    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    return n_components

@ray.remote
def get_phase_diagram(simplices,coords):
    num_comps = [label_simplex.remote(triangle,coords) for triangle in simplices]
    num_comps = ray.get(num_comps)

    return num_comps

def plot_phase_diagram(simplices,num_comps,coords,info):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    poly_coords = np.array([[0.0,0.0],[1.0,0.0],[0.5,np.sqrt(3)/2+0.01]])
    ax.add_patch(plt.Polygon(poly_coords,edgecolor=None, facecolor='brown',fill=True))
    words = [r'$\varphi_{p1}$',r'$\varphi_{p2}$',r'$\varphi_{s}$']
    xs = [-0.15,1,0.5]
    ys = [0,0,np.sqrt(3)/2+0.01]
    for x, y, s in zip(xs,ys,words):
        plt.text(x,y,s,fontsize=20)
    for i, triangle in zip(num_comps,simplices):
        tri_coords = coords[triangle,:]
        if i==2:
            ax.add_patch(plt.Polygon(tri_coords, edgecolor=None,facecolor='orange',fill=True))
        elif i==3:
            ax.add_patch(plt.Polygon(tri_coords,edgecolor=None, facecolor='lightblue',fill=True))

    ax.set_title(info['params'],pad=20)
    plt.axis('off')
    plt.legend(['1-phase','2-phase','3-phase'])
    plt.savefig('./'+info['fname']+'.png',dpi=500,bbox_inches='tight')

import time
since = time.time()
M,CHI,info=get_data(name='FHPaper',fhid=4)
grid, energy, hull,coords = ray.get(setup_data.remote(M,CHI,meshsize=200))
simplices = ray.get(refine_simplices.remote(hull,grid))
num_comps = ray.get(get_phase_diagram.remote(simplices,coords))
plot_phase_diagram(simplices,num_comps,coords,info)
print('Total elapsed time is: {}'.format(time.time()-since))





