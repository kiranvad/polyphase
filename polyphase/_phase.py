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
from itertools import combinations
from math import pi
from collections import defaultdict 
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

def label_simplex(grid, simplex, thresh):
    """ given a simplex, labels it to be a n-phase region by computing number of connected components """
    coords = [grid[:,x] for x in simplex]
    dist = squareform(pdist(coords,'euclidean'))
    adjacency = dist<thresh
    adjacency =  adjacency.astype(int)  
    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    return n_components

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
   
def is_boundary_point(point, zero_value = MIN_POINT_PRECISION):
    if np.isclose(point, MIN_POINT_PRECISION).any():
        return True
    else:
        return False

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
def _serialcompute(f, dimension, meshsize,**kwargs):
    """
    Main python function to obtain a phase diagram for n-component polymer mixture system.
    
    parameters:
    -----------
        f             : Energy function that take a composition and returns a scalar value
                        forexample : phase.polynomial_energy or lambda x : phase.flory_huggins(x, M,chi,beta=0.0)
        dimension     : number of components in the material system (int)                
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
        lift_grid_size               : A uniform grid to which simplex labels will be lifted to points inside corresponding simplices
        
    
    Output:
    -------
    A dictonary with the following keys:
        grid                         : Uniform grid used for discrete energy computation
        energy                       : (corrected) energy at each discrete point of the 'grid'
        thresh                       : Numerical distance threshold used for graph generation to compute number of connected components
        simplices                    : Simplices of convex hull of the energy landscape
        num_comps                    : Number of connected components of each simplex in 'simplices'
        output                       : A pandas dataframe with the rows corresponding to volume fractions (x dimensions) and a point phase                                          label computed using the lifting method 
                                       (if label=0 corresponding grid point is either does not belong to any simplex 
                                       or it lies on a collinear simplex) 
          
    """
    verbose = kwargs.get('verbose', False)
    flag_refine_simplices = kwargs.get('flag_refine_simplices', True)
    flag_lift_label = kwargs.get('flag_lift_label',False)
    use_weighted_delaunay = kwargs.get('use_weighted_delaunay', False)
    lift_grid_size = kwargs.get('lift_grid_size', meshsize)
                                           
    since = time.time()
    
    outdict = defaultdict(list)
    thresh_epsilon = 5e-3
    
    """ Perform a parallel computation of phase diagram """
    # 1. generate grid
    grid = makegridnd(meshsize, dimension)
    outdict['grid'] = grid
    
    lap = time.time()
    if verbose:
        print('{}-dimensional grid generated at {:.2f}s'.format(dimension,lap-since))

    # 2. compute free energy on the grid (parallel)
    
    flag_make_energy_paraboloid = kwargs.get('flag_make_energy_paraboloid',True)
    flag_lift_purecomp_energy = kwargs.get('flag_lift_purecomp_energy',False)
    
    if np.logical_or(flag_make_energy_paraboloid, flag_lift_purecomp_energy):
        beta = 0.0
    else:
        beta = kwargs.get('beta',1e-4)
        if verbose:
            print('Using beta (={:.2E}) correction for energy landscape'.format(beta))
         
    energy = np.asarray([f(x) for x in grid.T])

    lap = time.time()
    if verbose:
        print('Energy computed at {:.2f}s'.format(lap-since))
    
    max_energy = np.max(energy)
    # Make energy a paraboloid like by extending the landscape at the borders
    if flag_make_energy_paraboloid:
        pad_energy = kwargs.get('pad_energy',2)
        if verbose:
            print('Making energy manifold a paraboloid with {:d}x'
                  ' padding of {:.2f} maximum energy'.format(pad_energy, max_energy))
        boundary_points= np.asarray([is_boundary_point(x) for x in grid.T])
        energy[boundary_points] = pad_energy*max_energy         
    elif flag_lift_purecomp_energy:
        pad_energy = kwargs.get('pad_energy',2)
        if verbose:
            print('Aplpying {:d}x padding of {:.2f} maximum energy to pure components'.format(pad_energy, max_energy))
        pure_points = np.asarray([is_pure_component(x) for x in grid.T])
        energy[pure_points] = pad_energy*max_energy
    
    outdict['energy'] = energy
    
    lap = time.time()
    if verbose:
        print('Energy is corrected at {:.2f}s'.format(lap-since))
    
    # 3. Compute convex hull
    if not use_weighted_delaunay:
        _method = 'Convexhull'
        points = np.concatenate((grid[:-1,:].T,energy.reshape(-1,1)),axis=1)                   
        hull = ConvexHull(points)
        outdict['hull'] = hull
    else:
        raise NotImplemented
        
    lap = time.time()
    if verbose:
        print('{} is computed at {:.2f}s'.format(_method,lap-since))
    
    if not flag_refine_simplices:
        simplices = hull.simplices
    else:
        upper_hull = np.asarray([is_upper_hull(grid,simplex) for simplex in hull.simplices])
        simplices = hull.simplices[~upper_hull]
        outdict['upper_hull']=upper_hull
 
    lap = time.time()
    if verbose:
        print('Simplices are refined at {:.2f}s'.format(lap-since))
        
    outdict['simplices'] = simplices
    if verbose:
        print('Total of {} simplices in the convex hull'.format(len(simplices)))
        
    # determine threshold
    threshold_type = kwargs.get('threshold_type','delaunay')
    if threshold_type=='delaunay':
        thresh = get_max_delaunay_edge_length(grid) + thresh_epsilon
    elif threshold_type=='uniform':
        thresh_scale = kwargs.get('thresh_scale',1.25)
        thresh = thresh_scale*euclidean(grid[:,0],grid[:,1])
    
    if verbose:
        print('Using {:.2E} as a threshold for Laplacian of a simplex'.format(thresh)) 
        
    outdict['thresh'] = thresh
    
    # 4. for each simplex in the hull compute number of connected components (parallel)
    num_comps = [label_simplex(grid, simplex, thresh) for simplex in simplices]
    lap = time.time()
    if verbose:
        print('Simplices are labelled at {:.2f}s'.format(lap-since))
    outdict['num_comps'] = num_comps
    outdict['coplanar'] = None
    
    if flag_lift_label:
        
        # 5. lift the labels from simplices to points (parallel)
        if lift_grid_size == meshsize:
            lift_grid = grid
        else:
            lift_grid = makegridnd(lift_grid_size, dimensions) # we lift labels to a constant mesh 
            
        inside = [lift_label(grid, lift_grid, simplex, label) for simplex, label in zip(simplices, num_comps)]
        
        coplanar = [item[1] for item in inside]
        outdict['coplanar']=coplanar
        lap = time.time()
        if verbose:
            print('Labels are lifted at {:.2f}s'.format(lap-since))

            print('Total {}/{} coplanar simplices'.format(Counter(coplanar)[0],len(simplices)))

        phase = np.zeros(lift_grid.shape[1])
        for i,label in zip(inside,num_comps):
            if i[1]==1:
                phase[i[0]] = label
        phase = phase.reshape(1,-1)
        output = np.vstack((lift_grid,phase))
        index = ['Phi_'+str(i) for i in range(1, output.shape[0])]
        index.append('label')
        output = pd.DataFrame(data = output,index=index)
                
    else:
        output = []
        
    outdict['output'] = output 
    
    lap = time.time()
    print('Computation took {:.2f}s'.format(lap-since))
    
    return outdict

@ray.remote
def ray_is_boundary_point(point, zero_value = MIN_POINT_PRECISION):
    if np.isclose(point, MIN_POINT_PRECISION).any():
        return True
    else:
        return False

@ray.remote
def ray_is_pure_component(point, zero_value = MIN_POINT_PRECISION):
    counts = Counter(point)
    if counts[MIN_POINT_PRECISION]>1:
        return True
    else:
        return False

@ray.remote
def ray_label_simplex(grid, simplex, thresh):
    """ given a simplex, labels it to be a n-phase region by computing number of connected components """
    coords = [grid[:,x] for x in simplex]
    dist = squareform(pdist(coords,'euclidean'))
    adjacency = dist<thresh
    adjacency =  adjacency.astype(int)  
    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    return n_components

@ray.remote
def ray_is_upper_hull(grid, simplex):
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
def ray_lift_label(grid,lift_grid, simplex, label):
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

def _parcompute(f, dimension, meshsize,**kwargs):
    """Compute phase diagram using parallel computaion
    parallel version of serialcompute
    """
    verbose = kwargs.get('verbose', False)
    flag_refine_simplices = kwargs.get('flag_refine_simplices', True)
    flag_lift_label = kwargs.get('flag_lift_label',False)
    use_weighted_delaunay = kwargs.get('use_weighted_delaunay', False)
    lift_grid_size = kwargs.get('lift_grid_size', 200)
        
    # Initialize ray for parallel computation
    ray.init(ignore_reinit_error=True)

    since = time.time()
    
    outdict = {}
    thresh_epsilon = 5e-3
    """ Perform a parallel computation of phase diagram """
    # 1. generate grid
    grid = makegridnd(meshsize, dimension)
    outdict['grid'] = grid
    grid_ray = ray.put(grid)
    lap = time.time()
    if verbose:
        print('{}-dimensional grid generated at {:.2f}s'.format(dimension,lap-since))
        
    # 2. compute free energy on the grid (parallel)
    flag_make_energy_paraboloid = kwargs.get('flag_make_energy_paraboloid',True)
    flag_lift_purecomp_energy = kwargs.get('flag_lift_purecomp_energy',False)
    
    if np.logical_or(flag_make_energy_paraboloid, flag_lift_purecomp_energy):
        beta = 0.0
    else:
        beta = kwargs.get('beta',1e-4)
        if verbose:
            print('Using beta (={:.2E}) correction for energy landscape'.format(beta))   
            
    energy = np.asarray([f(x) for x in grid.T])   
    
    lap = time.time()
    if verbose:
        print('Energy computed at {:.2f}s'.format(lap-since))
        
    # Make energy a paraboloid like by extending the landscape at the borders
    if flag_make_energy_paraboloid:
        max_energy = np.max(energy)
        pad_energy = kwargs.get('pad_energy',2)
        if verbose:
            print('Making energy manifold a paraboloid with {:d}x padding of'
                  ' {:.2f} maximum energy'.format(pad_energy, max_energy))
        boundary_points_ray = [ray_is_boundary_point.remote(x) for x in grid.T]
        boundary_points = np.asarray(ray.get(boundary_points_ray))
        energy[boundary_points] = pad_energy*max_energy
        
        del boundary_points_ray
        
    elif flag_lift_purecomp_energy:
        max_energy = np.max(energy)
        pad_energy = kwargs.get('pad_energy',2)
        if verbose:
            print('Aplpying {:d}x padding of {:.2f} maximum energy to pure components'.format(pad_energy, max_energy))
        pure_points_ray = [ray_is_pure_component.remote(x) for x in grid.T]
        pure_points = np.asarray(ray.get(pure_points_ray))
        energy[pure_points] = pad_energy*max_energy
        
        del pure_points_ray
        
    outdict['energy'] = energy
    
    lap = time.time()
    if verbose:
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
    if verbose:
        print('{} is computed at {:.2f}s'.format(_method,lap-since))
    
    # determine threshold
    threshold_type = kwargs.get('threshold_type','delaunay')
    if threshold_type=='delaunay':
        thresh = get_max_delaunay_edge_length(grid) + thresh_epsilon
    elif threshold_type=='uniform':
        thresh_scale = kwargs.get('thresh_scale',1.25)
        thresh = thresh_scale*euclidean(grid[:,0],grid[:,1])
    
    if verbose:
        print('Using {:.2E} as a threshold for Laplacian of a simplex'.format(thresh)) 
        
    outdict['thresh'] = thresh
    
    if not flag_refine_simplices:
        simplices = hull.simplices
    else:
        upper_hull_ray = [ray_is_upper_hull.remote(grid_ray,simplex) for simplex in hull.simplices]
        upper_hull = np.asarray(ray.get(upper_hull_ray))
        simplices = hull.simplices[~upper_hull]
        outdict['upper_hull'] = upper_hull
        del upper_hull_ray

    outdict['simplices'] = simplices
    if verbose:
        print('Total of {} simplices in the convex hull'.format(len(simplices)))
    
    lap = time.time()
    if verbose:
        print('Simplices are refined at {:.2f}s'.format(lap-since))
    # 4. for each simplex in the hull compute number of connected components (parallel)
    num_comps_ray = [ray_label_simplex.remote(grid_ray, simplex, thresh) for simplex in simplices]
    num_comps = ray.get(num_comps_ray) 
    lap = time.time()
    if verbose:
        print('Simplices are labelled at {:.2f}s'.format(lap-since))
        
    outdict['num_comps'] = num_comps
    
    del num_comps_ray
    outdict['coplanar'] = None
    if flag_lift_label:
        
        # 5. lift the labels from simplices to points (parallel)
        if lift_grid_size == meshsize:
            lift_grid_ray = grid_ray
            lift_grid = grid
        else:
            lift_grid = makegridnd(lift_grid_size, dimensions) # we lift labels to a constant mesh 
            lift_grid_ray = ray.put(lift_grid)
            
        inside_ray = [ray_lift_label.remote(grid_ray, lift_grid_ray,
                                            simplex, label) for simplex, label in zip(simplices, num_comps)]
        inside = ray.get(inside_ray)
        
        coplanar = [item[1] for item in inside]
        outdict['coplanar'] = None
        lap = time.time()
        
        if verbose:
            print('Labels are lifted at {:.2f}s'.format(lap-since))

            print('Total {}/{} coplanar simplices'.format(Counter(coplanar)[0],len(simplices)))

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
    ray.shutdown()

    return outdict
