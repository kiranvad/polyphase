import pdb
import numpy as np
import time
import pandas as pd
import os
print('Number of cores available: {}'.format(os.cpu_count()))
from collections import Counter
    
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist, euclidean, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from numpy.linalg import norm

import warnings
from collections import Counter
from itertools import combinations

import ray
   
MIN_POINT_PRECISION = 1e-8


class WeightedDelaunay(object):
    """
    Given d-dimensional points as a numpy array of shape (n, d) and weights as a numpy array of shape (n,1),
    compute a weighted delaunay triangulations of points.
    
    Parameters:
    -----------
        points   :  d-dimensional points of shape (n, d)
        weights  :  weights for each point of shape (n,1)
        
        **kwargs :  inputs to scipy.spatial.ConvexHull
        
    Attributes:
    -----------
        points      :  d-dimensional points used for weighted delaunay construction of shape (n-points,d)
        vertices    :  vertices of weighted delaunay tesselations result (n-vertices, d)
        simplices   :  list of simplices in the weighted delaunay
        equations   :  normal vectors for each facet of shape (n-facets, d+1)    
    """
    def __init__(self, points, weights, **kwargs):
        self.num, self.dim = points.shape
        self.points = points
        self.weights = weights
        self.lifted_points = self.lift_to_paraboloid()

        self.conv = ConvexHull(self.lifted_points, **kwargs)
        self.centroid = np.mean(self.lifted_points[self.conv.vertices,:], axis=0)
        self.underside_facets = np.asarray([self._is_infsimplex(s) for s in self.conv.simplices])
        self.simplices = self.conv.simplices[~self.underside_facets]
        self.equations = self.conv.equations[~self.underside_facets]
        
        self.vertices = []
        for simplex in self.simplices:
            for v in simplex:
                self.vertices.append(self.points[v])
        self.vertices = np.asarray(self.vertices)
        
    def lift_to_paraboloid(self):
        """ 
        Use the lifting transform from scipy.spatial.Delaunay
        for lifting the points to paraboloid and weigh them
        """
        tri = Delaunay(self.points)
        z = np.asarray([tri.lift_points(x) for x in self.points])
        z[...,-1] = z[...,-1] - self.weights
        
        plane_centroid = np.mean(z[...,:-1], axis=0)
        
        self.pinf = np.append(plane_centroid,-1e10)
        z = np.vstack((z, self.pinf))
        
        return z
    
    def _is_downward_facing(self, normal):
        """ Return True if a facet is downward facing wrto centroid of the hull """
        value = np.dot(normal, np.hstack((self.centroid,1.0)))
        if value>0:
            flag = False
        elif value<0:
            flag = True
        else:
            raise RuntimeError('Dot product should be a real number!')
            
        return flag
    
    def _is_infsimplex(self, simplex):
        """ returns True if the simplex contains point at the infinity """
        if self.num in simplex:
            flag = True
        else:
            flag = False

        return flag
            
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

""" Main functions in remote setting """
@ray.remote
def label_simplex(grid, simplex, thresh):
    """ given a simplex, labels it to be a n-phase region by computing number of connected components """
    coords = [grid[:,x] for x in simplex]
    dist = squareform(pdist(coords,'euclidean'))
    adjacency = dist<thresh
    adjacency =  adjacency.astype(int)  
    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    return n_components

@ray.remote
def is_upper_hull(grid, simplex):
    """ 
    return True if a simplex connects anything on the edge.
    
    The assumption is that everything that connects to the edge belongs to upper convex hull.
    We would want to compute only the lower convex hull.
    """
    point = grid[:,simplex]
    if np.isclose(point, MIN_POINT_PRECISION).any():
        return True
    else:
        return False

@ray.remote    
def lift_label(grid,lift_grid, simplex, label):
    """ Lifting the labels from simplices to points """
    try:
        v = np.asarray([grid[:-1,x] for x in simplex])
        #inside = inpolyhedron(v, grid[:-1,:].T)
        tri = Delaunay(v)
        inside = Delaunay.find_simplex(tri,lift_grid[:-1,:].T)
        inside =~(inside<0)
        flag = 1
    except:
        inside = None
        flag = 0
        
    return inside, flag

"""Some helper functions"""
def flory_huggins(x, M,CHI,beta=1e-3):
    """ Free energy formulation """
    T1 = 0
    for i,xi in enumerate(x):
        T1 += (xi*np.log(xi))/M[i] + beta/xi
    T2 = 0.5*np.matmul((np.matmul(x,CHI)),np.transpose(x)) 
    
    return T1+T2  
        
# determine points inside a polyhedron
from scipy.spatial import Delaunay

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

@ray.remote
def is_collinear(grid,tri_coords, simplex):
    """ 
    determines whether a simplex is coplanar when lifted to design space.
    Returns True is a simplex has the lifted coordinates as collinear.
    """
    coords = np.array([tri_coords[x,:] for x in simplex])
    M= np.vstack((coords.T,np.array([1,1,1])))
    area = np.linalg.det(M)    
    flag = area<1e-15
    
    return flag

from math import pi
def get_ternary_coords(point):
    a,b,c = point
    x = 0.5-a*np.cos(pi/3)+b/2;
    y = 0.866-a*np.sin(pi/3)-b*(1/np.tan(pi/6)/2);
    
    return [x,y]

@ray.remote
def is_boundary_point(point, zero_value = MIN_POINT_PRECISION):
    if np.isclose(point, MIN_POINT_PRECISION).any():
        return True
    else:
        return False

@ray.remote
def is_pure_component(point, zero_value = MIN_POINT_PRECISION):
    counts = Counter(point)
    if counts[MIN_POINT_PRECISION]>1:
        return True
    else:
        return False

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
            
            
""" Main comoutation function """
def compute(dimensions, configuration, meshsize,\
            flag_refine_simplices = True, flag_lift_label =False,\
            lift_grid_size=200, use_weighted_delaunay = False, **kwargs):
    """
    Main python function to obtain a phase diagram for n-component polymer mixture system.
    
    parameters:
    -----------
        dimension     : number of components of mixture
        configuration : a dictornay with keys:
                            'M'   : degree of polymerization (list of length = dimension)
                            'chi' : off diagonal non-zero entries of flory-huggins parameters
                                    (exmaple: three component system : [chi_12, chi_13, chi_23])
        meshsize      : number of grid points per dimension  
        
        kwargs:
        -------
        flag_refine_simplices        : whether to remove simplices that connect pure components (default: True)
        flag_lift_label              : Whether to list labels of simplices to a point cloud of constant size (default: False). Point cloud                                          is constantsize regadless of original meshsize and is computed using a 200 points per dimension mesh.         use_weighted_delaunay        : Uses a weighted delaunay triangulation to compute triangulation of design space. (not complete yet)
        beta                         : beta correction value used to compute flory-huggins free energy (default 1e-4)
        flag_remove_collinear        : In three dimensional case, removes simplices that lift to collinear points in phi space. 
                                       (default, False)
        flag_make_energy_paraboloid  : Bypasses the beta correction and makes the boundary points to have a constant energy (default, True)
        pad_energy                   : factor of maximum energy used as a padding near the boundary of manifold. (default, 2)
        flag_lift_purecomp_energy    : Makes the energy of pure components `pad_energy` times the maximum energy
        threshold_type               : Whether to use an 'uniform' threshold method (thresh= edge length) or to use more mathematically                                            sound 'delaunay' (thresh = maximum delaunay edge + epsilon)
        thresh_scale                 : (scale value of) Uniform edge length threshold to compute adjacency matrix 
                                       (default: 1.25 times the uniform edge in the grid)
        
    """
    
    # Initialize ray for parallel computation
    #ray.init(ignore_reinit_error=True, lru_evict=False)

    since = time.time()
    
    outdict = {}
    thresh_epsilon = 5e-3
    """ Perform a parallel computation of phase diagram """
    # 1. generate grid
    grid = makegridnd(meshsize, dimensions)
    outdict['grid'] = grid
    grid_ray = ray.put(grid)
    lap = time.time()
    print('{}-dimensional grid generated at {:.2f}s'.format(dimensions,lap-since))

    # 2. compute free energy on the grid (parallel)
    CHI = _utri2mat(configuration['chi'], dimensions)
    flag_make_energy_paraboloid = kwargs.get('flag_make_energy_paraboloid',True)
    flag_lift_purecomp_energy = kwargs.get('flag_lift_purecomp_energy',False)
    
    if np.logical_or(flag_make_energy_paraboloid, flag_lift_purecomp_energy):
        beta = 0.0
    else:
        beta = kwargs.get('beta',1e-4)
        
    energy = np.asarray([flory_huggins(x,configuration['M'],CHI,beta=beta) for x in grid.T])    
    lap = time.time()
    print('Energy computed at {:.2f}s'.format(lap-since))
    # Make energy a paraboloid like by extending the landscape at the borders
    if flag_make_energy_paraboloid:
        max_energy = np.max(energy)
        pad_energy = kwargs.get('pad_energy',2)
        print('Making energy manifold a paraboloid with {:d}x padding of {:.2f} maximum energy'.format(pad_energy, max_energy))
        boundary_points_ray = [is_boundary_point.remote(x) for x in grid.T]
        boundary_points = np.asarray(ray.get(boundary_points_ray))
        energy[boundary_points] = pad_energy*max_energy
        
        del boundary_points_ray
        
    elif flag_lift_purecomp_energy:
        max_energy = np.max(energy)
        pad_energy = kwargs.get('pad_energy',2)
        print('Aplpying {:d}x padding of {:.2f} maximum energy to pure components'.format(pad_energy, max_energy))
        pure_points_ray = [is_pure_component.remote(x) for x in grid.T]
        pure_points = np.asarray(ray.get(pure_points_ray))
        energy[pure_points] = pad_energy*max_energy
        
        del pure_points_ray
        
    outdict['energy'] = energy
    
    lap = time.time()
    print('Energy is corrected at {:.2f}s'.format(lap-since))
    
    # 3. Compute convex hull
    if not use_weighted_delaunay:
        _method = 'Convexhull'
        points = np.concatenate((grid[:-1,:].T,energy.reshape(-1,1)),axis=1)
        hull = ConvexHull(points)
        outdict['hull'] = hull
    else:
        _method = 'Weighted Delaunay'
        points = grid[:-1,:].T
        weights = energy.reshape(-1)
        hull = WeightedDelaunay(points, weights)
        outdict['hull'] = hull
        
    lap = time.time()
    print('{} is computed at {:.2f}s'.format(_method,lap-since))
    
    # determine threshold
    threshold_type = kwargs.get('threshold_type','delaunay')
    if threshold_type=='delaunay':
        thresh = get_max_delaunay_edge_length(grid) + thresh_epsilon
    elif threshold_type=='uniform':
        thresh_scale = kwargs.get('thresh_scale',1.25)
        thresh = thresh_scale*euclidean(grid[:,0],grid[:,1])
    
    print('Using {:.2E} as a threshold for Laplacian of a simplex'.format(thresh)) 
    outdict['thresh'] = thresh
    
    if not flag_refine_simplices:
        simplices = hull.simplices
    else:
        upper_hull_ray = [is_upper_hull.remote(grid_ray,simplex) for simplex in hull.simplices]
        upper_hull = np.asarray(ray.get(upper_hull_ray))
        simplices = hull.simplices[~upper_hull]
        
        del upper_hull_ray
    
    flag_remove_collinear = kwargs.get('flag_remove_collinear',False)
    if flag_remove_collinear :
        # remove colpanar simplices
        tri_coords_ray = ray.put(np.array([get_ternary_coords(pt) for pt in grid.T]))
        coplanar_ray = [is_collinear.remote(grid_ray,tri_coords_ray,simplex) for simplex in simplices]
        coplanar_simplices = np.asarray(ray.get(coplanar_ray))
        if len(coplanar_simplices)==0:
            warnings.warn('There are no coplanar simplices.')
        else:    
            simplices = simplices[~coplanar_simplices]
            
        lap = time.time()
        print('Simplices are refined at {:.2f}s'.format(lap-since))

        del tri_coords_ray, coplanar_ray
        
    outdict['simplices'] = simplices
    print('Total of {} simplices in the convex hull'.format(len(simplices)))
    
    # 4. for each simplex in the hull compute number of connected components (parallel)
    num_comps_ray = [label_simplex.remote(grid_ray, simplex, thresh) for simplex in simplices]
    num_comps = ray.get(num_comps_ray) 
    lap = time.time()
    print('Simplices are labelled at {:.2f}s'.format(lap-since))
    outdict['num_comps'] = num_comps
    
    del num_comps_ray
    
    if flag_lift_label:
        
        # 5. lift the labels from simplices to points (parallel)
        if lift_grid_size == meshsize:
            lift_grid_ray = grid_ray
            lift_grid = grid
        else:
            lift_grid = makegridnd(lift_grid_size, dimensions) # we lift labels to a constant mesh 
            lift_grid_ray = ray.put(lift_grid)
            
        inside_ray = [lift_label.remote(grid_ray, lift_grid_ray, simplex, label) for simplex, label in zip(simplices, num_comps)]
        inside = ray.get(inside_ray)
        
        flags = [item[1] for item in inside]
        lap = time.time()
        print('Labels are lifted at {:.2f}s'.format(lap-since))
        
        print('Total {}/{} coplanar simplices'.format(Counter(flags)[0],len(simplices)))

        phase = np.zeros(lift_grid.shape[1])
        for i,label in zip(inside,num_comps):
            if i[1]==1:
                phase[i[0]] = label
        phase = phase.reshape(1,-1)
        output = np.vstack((lift_grid,phase))
        index = ['Phi_'+str(i) for i in range(1, output.shape[0])]
        index.append('label')
        output = pd.DataFrame(data = output,index=index)
        
        del lift_grid_ray, inside_ray, inside
        
    else:
        output = []
        
    outdict['output'] = output    
    lap = time.time()
    print('Computation took {:.2f}s'.format(lap-since))
    
    # we remove everything we don't need
    del grid_ray  
    
    # finish computation and exit ray
    #ray.shutdown()

    return outdict

if __name__ == '__main__':
    """ configure your material system """
    dimensions = 3
    M = np.ones(dimensions)
    chi = 3.10*np.ones(int(0.5*dimensions*(dimensions-1)))
    configuration = {'M': M, 'chi':chi}
    dx = 200
    output, simplices, grid, num_comps = compute(dimensions, configuration, dx)
    filepath = "./output"
    output.to_csv(filepath + ".csv")

    """ Post-processing """
    # 6. plotting the phase diagram 
    import sys
    if '../' not in sys.path:
        sys.path.insert(0,'../')
    import matplotlib.pyplot as plt

    if dimensions==3:
        from solvers.visuals import plot_3d_phasediagram
        plot_3d_phasediagram(grid, simplices, num_comps)
        fname = filepath + ".png"
        plt.savefig(fname, dpi=500, bbox_inches = "tight")
        plt.close()
    
    
    
    