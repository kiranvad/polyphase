import pdb
import numpy as np
import time
import pandas as pd
import os
from collections import Counter
    
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist, euclidean, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from numpy.linalg import norm

import warnings
from collections import Counter
from itertools import combinations
from math import pi

import ray

MIN_POINT_PRECISION = 1e-8
            
""" Main functions serial """
def makegridnd(meshsize, dimension):
    """
    Given mesh size and a dimensions, creates a n-dimensional grid for the volume fraction.
    Note that the grid would be a hyper plane in the n-dimensions.
    """
    x = np.meshgrid(*[np.linspace(MIN_POINT_PRECISION, 1,meshsize) for d in range(dimension)])
    mesh = np.asarray(x)
    total = np.sum(mesh,axis=0)
    plane_mesh = mesh[:,np.isclose(total,1.0,atol=1e-2)]

    return plane_mesh

def _utri2mat(utri, dimension):
    """ convert list of chi values to a matrix form """
    inds = np.triu_indices(dimension,1)
    ret = np.zeros((dimension, dimension))
    ret[inds] = utri
    ret.T[inds] = utri

    return ret

"""Some helper functions"""
def flory_huggins(x, M,CHI,beta=1e-3):
    """ Free energy formulation """
    T1 = 0
    for i,xi in enumerate(x):
        T1 += (xi*np.log(xi))/M[i] + beta/xi
    T2 = 0.5*np.matmul((np.matmul(x,CHI)),np.transpose(x)) 
    
    return T1+T2  
        
def polynomial_energy(x):
    """ Free energy using a polynomial function for ternary """
    
    assert len(x)==3,'Expected a ternary system got {}'.format(len(x))
    
    #e = (x[0]**2)*(x[1]**2) + (x[0]**2 + x[1]**2)*(x[2]**2)
    # e = -e/0.5
    e =0
    for xi in x:
        e += ((xi-0.1)**2)*((0.9-xi)**2)

    return e*1e3
    
def inpolyhedron(ph,points):
    """
    Given a polyhedron vertices in `ph`, and `points` return 
    critera that each point is either with in or outside the polyhedron
    
    Both polyhedron and points should have the same shape i.e. num_points X num_dimensions
    
    Returns a boolian array : True if inside, False if outside
    
    """
    tri = Delaunay(ph)
    inside = Delaunay.find_simplex(tri,points)
    criteria = inside<0
    return ~criteria

def get_max_delaunay_edge_length(grid):
    delaunay = Delaunay(np.asarray(grid[:-1,:].T))
    max_delaunay_edge = 0.0
    for sx in delaunay.simplices:
        vertex_sx = [grid[:,x] for x in sx]
        edges = combinations(vertex_sx, 2)
        edge_lengths = np.array([norm(e[0]-e[1]) for e in edges])
        current_max = np.max(edge_lengths)
        if max_delaunay_edge<current_max:
            max_delaunay_edge = current_max
    
    return max_delaunay_edge       
            

class PHASE:
    def __init__(self,energy_func, meshsize,dimension):
        if not callable(energy_func):
            raise ValueError('Vairable energy needs to be a function such as utils.flory_huggins')
        self.energy_func = energy_func
        self.meshsize = meshsize
        self.dimension = dimension
        self.grid = self.makegridnd()
    
    def makegridnd(self):
        """
        Given mesh size and a dimensions, creates a n-dimensional grid for the volume fraction.
        Note that the grid would be a hyper plane in the n-dimensions.
        """
        x = np.meshgrid(*[np.linspace(MIN_POINT_PRECISION, 1,self.meshsize) for d in range(self.dimension)])
        mesh = np.asarray(x)
        total = np.sum(mesh,axis=0)
        plane_mesh = mesh[:,np.isclose(total,1.0,atol=1e-2)]

        return plane_mesh
    
    def label_simplex(self, simplex):
        """Produce simplex phase labels
        given a simplex, labels it to be a n-phase region by computing number of connected components 
        """
        coords = [self.grid[:,x] for x in simplex]
        dist = squareform(pdist(coords,'euclidean'))
        adjacency = dist<self.thresh
        adjacency =  adjacency.astype(int)  
        graph = csr_matrix(adjacency)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

        return n_components
    
    def is_upper_hull(self, simplex):
        """Identify simplices in the upper hull
        return True if a simplex connects anything on the edge.

        The assumption is that everything that connects to the edge belongs to upper convex hull.
        We would want to compute only the lower convex hull.
        """
        point = self.grid[:,simplex]
        if np.isclose(point, MIN_POINT_PRECISION).any():
            return True
        else:
            return False
        
    def simplex2points_label(self, simplex, label):
        """ Lifting the labels from simplices to points """
        try:
            v = np.asarray([self.grid[:-1,x] for x in simplex])
            tri = Delaunay(v)
            inside = Delaunay.find_simplex(tri,self.grid[:-1,:].T)
            inside =~(inside<0)
            flag = 1
        except:
            inside = None
            flag = 0

        return inside, flag  
    
    def is_boundary_point(self,point):
        if np.isclose(point, MIN_POINT_PRECISION).any():
            return True
        else:
            return False

    def is_pure_component(self,point):
        counts = Counter(point)
        if counts[MIN_POINT_PRECISION]>1:
            return True
        else:
            return False
        
    def _process_parallel(self, f, iterator):
        @ray.remote
        def remote_func(indx, f_args):
            if callable(f):
                if isinstance(f_args, tuple):
                    f_out = f(*f_args)
                else:
                    f_out = f(f_args)
                    
                return f_out, indx

        remaining_result_ids  = [remote_func.remote(indx, f_args) for indx,f_args in enumerate(iterator)]
        
        if self.verbose:
            print('Total of {} remote functions for <{}>'.format(len(iterator), f.__name__))
            
        out=[None]*len(iterator)
        while len(remaining_result_ids) > 0:
            ready_result_ids, remaining_result_ids = ray.wait(remaining_result_ids, num_returns=1)
            results = ray.get(ready_result_ids)[0]
            out[results[1]] = results[0]

        del remaining_result_ids, ready_result_ids

        return out
    
    def solve(self,**kwargs):
        """Solve for phase diagram using convex envelope method
        
        """
        self.use_parallel = kwargs.get('use_parallel', False)
        if self.use_parallel:
            ray.init(ignore_reinit_error=True, local_mode=False)
            
        self.verbose = kwargs.get('verbose', False)
        self.correction = kwargs.get('correction', 'edge')
        self.lift_label = kwargs.get('lift_label',False)

        since = time.time()
        thresh_epsilon = 5e-3
        
        self.energy = np.asarray([self.energy_func(x) for x in self.grid.T])
        lap = time.time()
        if self.verbose:
            print('Energy computed at {:.2f}s'.format(lap-since))
        max_energy = np.max(self.energy)
        
        if self.correction=='edge':
            if self.verbose:
                print('Using edge correction to energy landscape')
                
            if self.use_parallel:
                boundary_points = np.asarray(self._process_parallel(self.is_boundary_point, self.grid.T), dtype=bool)
            else:
                boundary_points= np.asarray([self.is_boundary_point(x) for x in self.grid.T])
            
            self.energy[boundary_points] = 2*max_energy

        elif self.correction=='vertex':
            if self.verbose:
                print('Using vertex correction to energy landscape. THIS IS NOT RECOMMENDED.')
                
            if self.use_parallel:
                pure_points = np.asarray(self._process_parallel(self.is_pure_component, self.grid.T), dtype=bool)
            else:
                pure_points= np.asarray([self.is_pure_component(x) for x in self.grid.T]) 
                
            self.energy[pure_points] = 2*max_energy

        lap = time.time()
        if self.verbose:
            print('Energy is corrected at {:.2f}s'.format(lap-since))

        points = np.concatenate((self.grid[:-1,:].T,self.energy.reshape(-1,1)),axis=1)
        self.hull = ConvexHull(points)

        lap = time.time()
        if self.verbose:
            print('Convexhull is computed at {:.2f}s'.format(lap-since))

        self.thresholding = kwargs.get('thresholding','uniform')
        
        if self.thresholding=='delaunay':
            self.thresh = get_max_delaunay_edge_length(self.grid) + thresh_epsilon
        elif self.thresholding=='uniform':
            thresh_scale = kwargs.get('thresh_scale',1.25)
            self.thresh = thresh_scale*euclidean(self.grid[:,0],self.grid[:,1])

        if self.verbose:
            print('Using {:.2E} as a threshold for Laplacian of a simplex'.format(self.thresh)) 
            
        self.refine_simplices = kwargs.get('refine_simplices',True)
        if not self.refine_simplices:
            self.simplices = hull.simplices
        else:
            if self.use_parallel:
                self.upper_hull = np.asarray(self._process_parallel(self.is_upper_hull, self.hull.simplices), dtype=bool)
            else:
                self.upper_hull = np.asarray([self.is_upper_hull(simplex) for simplex in self.hull.simplices])
                
            self.simplices = self.hull.simplices[~self.upper_hull]

        if self.verbose:
            print('Total of {} simplices in the convex hull'.format(len(self.simplices)))

        if self.use_parallel:
            self.num_comps = self._process_parallel(self.label_simplex, self.simplices)
        else:
            self.num_comps = [self.label_simplex(simplex) for simplex in self.simplices]
        
        lap = time.time()
        if self.verbose:
            print('Simplices are labelled at {:.2f}s'.format(lap-since))

        if self.lift_label:
            if self.use_parallel:
                iterator = list(zip(self.simplices, self.num_comps))
                inside = self._process_parallel(self.simplex2points_label, iterator) 
            else:
                inside = [self.simplex2points_label(simplex, label) for simplex, label in zip(self.simplices, self.num_comps)]
            flags = [item[1] for item in inside]
            lap = time.time()
            if self.verbose:
                print('Labels are lifted at {:.2f}s'.format(lap-since))
                print('Total {}/{} coplanar simplices'.format(Counter(flags)[0],len(self.simplices)))

            phase = np.zeros(self.grid.shape[1])
            for i,label in zip(inside,self.num_comps):
                if i[1]==1:
                    phase[i[0]] = label
            phase = phase.reshape(1,-1)
            output = np.vstack((self.grid,phase))
            index = ['Phi_'+str(i) for i in range(1, output.shape[0])]
            index.append('label')
            self.df = pd.DataFrame(data = output,index=index)

        lap = time.time()
        print('Computation took {:.2f}s'.format(lap-since))
        
        if self.use_parallel:
            ray.shutdown()
        return 
