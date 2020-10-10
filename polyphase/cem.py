import numpy as np
from scipy.spatial import ConvexHull
import math 
from scipy.spatial.distance import cdist
from numpy.linalg import norm
import pdb

import sys
if '../' not in sys.path:
    sys.path.insert(0,'../')

from solvers.helpers import get_ternary_coords

class CEM:
    def __init__(self,grid, gmix, energy = None):
        """
        Implementation of Convex Envelope Method (CEM) for identification of common tangetns
        using convex envelope of a free energy landscape.
        
        Inputs;
        --------
        grid   :  Grid of points where free enery needs to be computed. A nd array.
        gmix   :  Free enrgy functional formation with a n-dimensional point as input
        
        ToDos:
        ------
        Currently works for 3D because of the conversion to ternary coordinate. 
        Fix it work for any arbitrary n-dimension
        
        """
        self.grid = grid
        if energy is None:
            energy,coords = [],[]
            for point in self.grid:
                energy.append(gmix(point))
                coords.append(get_ternary_coords(point))
            self.energy = energy
            self.coords = np.asarray(coords)
        else:
            self.energy = energy
            coords = []
            for point in grid:
                coords.append(get_ternary_coords(point))
                
            self.coords = np.asarray(coords)
        
    
    def _get_angles(self,A, B, C):  
        ba = A-B
        bc = C-B
        alpha = np.degrees(np.arccos(np.dot(ba,bc)/(norm(ba)*norm(bc)+1e-4)))
        ca = A-C
        cb = B-C
        beta = np.degrees(np.arccos(np.dot(ca,cb)/(norm(ca)*norm(cb)+1e-4)))   

        gamma = 180-alpha-beta
    
        return np.round(alpha),np.round(beta),np.round(gamma)

    def _classify_triangles(self,A,B,C):
        alpha, beta, gamma = self._get_angles(A,B,C)
        criterion = len(np.unique([alpha,beta,gamma]))
        if criterion==1:
            flag = 0 # Equilateral
        elif criterion==2:
            flag = 1 # iscoeless
        else:
            flag = 2 # scalane

        return flag 
    
    def get_binodal(self):
        points = np.concatenate((self.coords,np.asarray(self.energy).reshape(-1,1)),axis=1)
        hull = ConvexHull(points)
        score = np.zeros(np.shape(self.grid))
        for simplex in hull.simplices:
            x,y,z = simplex[0],simplex[1],simplex[2]
            """
            A work around is to use the triangles in the volume fraction grid to classify
            """
            A,B,C = self.coords[x,:],self.coords[y,:],self.coords[z,:]
            # A,B,C = self.grid[x,:],self.grid[y,:],self.grid[z,:]

            flag = self._classify_triangles(A,B,C)
            score[x,flag] += 1
            score[y,flag] += 1
            score[z,flag] += 1
            
        score = np.argmax(score,axis=1)
        
        return score
    
    def _test_triangle_classification(self):
        A, B , C = np.array([0,0]),np.array([1,0]),np.array([0.5,np.sqrt(3)/2])
        assert self._classify_triangles(A,B,C)==0
        A, B , C = np.array([0,0]),np.array([1,0]),np.array([0,1])
        assert self._classify_triangles(A,B,C)==1
        A, B , C = np.array([0,0]),np.array([1,0]),np.array([-1,np.sqrt(3)/2])
        assert self._classify_triangles(A,B,C)==2

from scipy.spatial.distance import pdist, euclidean, squareform
from scipy.sparse import csr_matrix
import pandas as pd
from scipy.sparse.csgraph import connected_components        
        
class CEMLaplace(CEM):
    def __init__(self,grid, gmix, energy = None):
        super().__init__(grid, gmix, energy)
        points = np.concatenate((self.coords,np.asarray(self.energy).reshape(-1,1)),axis=1)
        self.hull = ConvexHull(points)
        
    def _compute_laplace_labelling(self):

        thresh = 5*euclidean(self.coords[0,:],self.coords[1,:])
        num_comp_list = []

        for triangle in self.hull.simplices:
            tri_coords = self.coords[triangle,:]
            dist = squareform(pdist(tri_coords,'euclidean'))
            adjacency = dist<thresh
            adjacency =  adjacency.astype(int)  
            graph = csr_matrix(adjacency)
            n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
            num_comp_list.append(n_components)
            
        return num_comp_list
    
    def plot_binodal(self,ax,fig,labels=None):
        num_comp_list = self._compute_laplace_labelling()
        ax.set_aspect('equal')
        tpc = ax.tripcolor(self.coords[:,0], self.coords[:,1], self.hull.simplices, \
                           facecolors=np.asarray(num_comp_list), edgecolors='k')
        cbar = fig.colorbar(tpc, ticks=[1, 2, 3])
        cbar.ax.set_yticklabels(['1-phase', '2-phase', '3-phase'])
        cbar.set_label('Phase region identification')
        if labels is None:
            words = [r'$\varphi_{p1}$',r'$\varphi_{p2}$',r'$\varphi_{s}$']
        else:
            words = labels
            
        xs = [-0.15,1,0.5]
        ys = [0,0,np.sqrt(3)/2+0.01]
        for x, y, s in zip(xs,ys,words):
            ax.text(x,y,s,fontsize=20)
        
        return ax
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        