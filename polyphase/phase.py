from matplotlib import rc
rc('text', usetex=True)
import seaborn as sns
import matplotlib.pyplot as plt

import pdb
import numpy as np
import time

import sys
if '../' not in sys.path:
    sys.path.append('../')

from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, euclidean, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from solvers import helpers
        
""" Main class for computing phase labels """    
class PhaseModelling(object):
    """
    Main python class used to obtain a phase diagram for n-component polymer mixture system.
    
    parameters:
    -----------
        dimension : number of components of mixture
        configuration : a dictornay with keys:
                        'M' : degree of polymerization (list of length = dimension)
                        'chi' : off diagonal non-zero entries of flory-huggins parameters
                                (exmaple: for two component system: [chi_12], three component system : [chi_12, chi_13, chi_23])
        meshsize  : number of grid points per dimension  
        refine_simplices  : weather to remove simplices that connect pure components
    
    attributes:
    -----------
        run : computes phase labels for the convex lifted simplex.
        plot : for three and four component system plots the phase diagram
    
    """
    
    def __init__(self,dimension,configuration, meshsize=40,refine_simplices=True):
        self.dimension = dimension
        self.M = configuration['M']
        self.CHI = self._utri2mat(configuration['chi'])
        self.meshsize = meshsize
        self.refine_simplices = refine_simplices
        
    def makegridnd(self):
        """
        Given mesh size and a dimensions, creates a n-dimensional grid for the volume fraction.
        Note that the grid would be a hyper plane in the n-dimensions.
        """
        x = np.meshgrid(*[np.linspace(0.001, 1,self.meshsize) for d in range(self.dimension)])
        mesh = np.asarray(x)
        total = np.sum(mesh,axis=0)
        plane_mesh = mesh[:,np.isclose(total,1.0,atol=1e-2)]

        return plane_mesh
    
    def _utri2mat(self,utri):
        """ convert list of chi values to a matrix form """
        inds = np.triu_indices(self.dimension,1)
        ret = np.zeros((self.dimension, self.dimension))
        ret[inds] = utri
        ret.T[inds] = utri
        
        return ret
    
    def _refine_simplices(self,hull):
        """ refine the simplices such that we only find a convex hull """
        simplices = []
        for simplex in hull.simplices:
            point_class = np.sum(np.isclose(self.grid[:,simplex],0.001),axis=1)
            point_class = np.unique(self.dimension - point_class)
            if (point_class==1).any():
                pass
            else:
                simplices.append(simplex)
                
        return simplices
    
    def label_simplex(self,simplex):
        """ given a simplex, labels it to be a n-phase region by computing number of connected components """
        coords = [self.grid[:,x] for x in simplex]
        dist = squareform(pdist(coords,'euclidean'))
        adjacency = dist<self.thresh
        adjacency =  adjacency.astype(int)  
        graph = csr_matrix(adjacency)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

        return n_components

    def get_phase_diagram(self):
        """ repeats the labelling for each simplex in the lifted simplicial complex """
        num_comps = [self.label_simplex(simplex) for simplex in self.simplices]

        return num_comps 
    
    def run(self):
        """ computes the phase labels for the n-component system class"""
        self.since = time.time()
        self.grid = self.makegridnd()
        lap = time.time()
        print('{}-dimensional grid geenrated at {:.2f}s'.format(self.dimension,lap-self.since))
        gmix = lambda x: helpers.flory_huggins(x, self.M, self.CHI,beta=1e-4)
        self.energy = []
        for i in range(self.grid.shape[1]):
            self.energy.append(gmix(self.grid[:,i]))
            

        points = np.concatenate((self.grid[:-1,:].T,np.asarray(self.energy).reshape(-1,1)),axis=1)
        hull = ConvexHull(points)
        lap = time.time()
        print('Convexhull is computed at {:.2f}s'.format(lap-self.since))
        
        self.thresh = 5*euclidean(self.grid[:,0],self.grid[:,1])
        
        if self.refine_simplices:
            self.simplices = self._refine_simplices(hull)
        else:
            self.simplices = hull.simplices
        self.num_comps = self.get_phase_diagram()
        lap = time.time()
        print('Simplices are labelled at {:.2f}s'.format(lap-self.since))
        
        if self.dimension==4:
            self._interpolate_labels_4d()
            
        
        return self.num_comps

    def plot(self,ax = None,**kwargs):
        """ a helper plotting function got 3 and 4 component phase diagrams"""
        
        if self.dimension ==3:
            ax, cbar = plot_3d_phasediagram(self,ax=ax)
        elif self.dimension == 4:
            sliceat = kwargs.pop('sliceat', 0.5)         
            ax, cbar = plot_4d_phasediagram(self,sliceat=sliceat,ax=ax,**kwargs)

        else:
            raise NotImplemented
        
        return ax,cbar
    
    def _interpolate_labels_4d(self):
        self.coords = np.asarray([helpers.from4d23d(point) for point in self.grid.T])
        self.phase = np.zeros(self.coords.shape[0])
        info = {'coplanar':[]}
        for i,simplex in zip(self.num_comps,self.simplices):
            try:
                v = np.asarray([self.grid[:-1,x] for x in simplex])
                inside = helpers.inpolyhedron(v, self.grid[:-1,:].T)
                self.phase[inside]=i
            except:
                info['coplanar'].append([i,simplex])
        lap = time.time()
        print('Labels are lifted at {:.2f}s'.format(lap-self.since))
        print('Total {}/{} coplanar simplices'.format(len(info['coplanar']),len(self.simplices)))
        
    
from matplotlib import colors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_4d_phasediagram(pm, sliceat=0.5, ax = None,**kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = plt.gcf()
        
    v = np.array([[0, 0, 0], [1, 0, 0], [1/2,np.sqrt(3)/2,0],  [1/2,np.sqrt(3)/6,np.sqrt(6)/3]])
    ax.scatter3D(v[:, 0], v[:, 1], v[:, 2],color='black')
    verts = helpers.get_convex_faces(v)
    ax.add_collection3d(Poly3DCollection(verts, facecolors='black', linewidths=0.5, edgecolors='black', alpha=.05))
    
    criteria = np.logical_and(pm.grid[3,:]<sliceat,pm.phase>0)
    cmap = colors.ListedColormap(['tab:red','tab:olive','tab:cyan','tab:purple'])
    boundaries = np.linspace(1,5,5)
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    surf = ax.scatter3D(pm.coords[criteria, 0], pm.coords[criteria, 1], pm.coords[criteria, 2],\
                        c=pm.phase[criteria],cmap=cmap,norm=norm)
    if kwargs.get('cbar', True) is True:
        cbar = fig.colorbar(surf, shrink=0.5, aspect=5, ticks=[1.5,2.5,3.5,4.5])
        cbar.ax.set_yticklabels(['1-Phase', '2-Phase', '3-Phase','4-Phase'])
    else:
        cbar = []
    words = [r'$\varphi_{1}$',r'$\varphi_{2}$',r'$\varphi_{3}$',r'$\varphi_{4}$']
    for vi,w in zip(v,words):
        ax.text(vi[0],vi[1],vi[2],w,fontsize=20)
    ax.set_axis_off()

    return ax, cbar
        
def plot_3d_phasediagram(pm, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    ax.set_aspect('equal')
    
    coords = np.asarray([helpers.get_ternary_coords(pt) for pt in pm.grid.T])
    tpc = ax.tripcolor(coords[:,0], coords[:,1], pm.simplices, facecolors=np.asarray(pm.num_comps), edgecolors='none')
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

    
    
    
    
    
    
    
    
    
    