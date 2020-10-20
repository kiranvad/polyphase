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

MIN_POINT_PRECISION = 1e-8

""" Main functions serial """


def makegridnd(meshsize, dimension):
    """
    Given mesh size and a dimensions, creates a n-dimensional grid for the volume fraction.
    Note that the grid would be a hyper plane in the n-dimensions.
    """
    x = np.meshgrid(*[np.linspace(MIN_POINT_PRECISION, 1, meshsize) for d in range(dimension)])
    mesh = np.asarray(x)
    total = np.sum(mesh, axis=0)
    plane_mesh = mesh[:, np.isclose(total, 1.0, atol=1e-2)]

    return plane_mesh


def _utri2mat(utri, dimension):
    """ convert list of chi values to a matrix form """
    inds = np.triu_indices(dimension, 1)
    ret = np.zeros((dimension, dimension))
    ret[inds] = utri
    ret.T[inds] = utri

    return ret


def label_simplex(grid, simplex, thresh):
    """ given a simplex, labels it to be a n-phase region by computing number of connected components """
    coords = [grid[:, x] for x in simplex]
    dist = squareform(pdist(coords, 'euclidean'))
    adjacency = dist < thresh
    adjacency = adjacency.astype(int)
    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    return n_components


def is_upper_hull(grid, simplex):
    """ 
    return True if a simplex connects anything on the edge.
    
    The assumption is that everything that connects to the edge belongs to upper convex hull.
    We would want to compute only the lower convex hull.
    """
    point = grid[:, simplex]
    if np.isclose(point, MIN_POINT_PRECISION).any():
        return True
    else:
        return False


def lift_label(grid, lift_grid, simplex, label):
    """ Lifting the labels from simplices to points """
    try:
        v = np.asarray([grid[:-1, x] for x in simplex])
        # inside = inpolyhedron(v, grid[:-1,:].T)
        tri = Delaunay(v)
        inside = Delaunay.find_simplex(tri, lift_grid[:-1, :].T)
        inside = ~(inside < 0)
        flag = 1
    except:
        inside = None
        flag = 0

    return inside, flag


"""Some helper functions"""


def flory_huggins(x, M, CHI, beta=1e-3):
    """ Free energy formulation """
    T1 = 0
    for i, xi in enumerate(x):
        T1 += (xi * np.log(xi)) / M[i] + beta / xi
    T2 = 0.5 * np.matmul((np.matmul(x, CHI)), np.transpose(x))

    return T1 + T2


# determine points inside a polyhedron
from scipy.spatial import Delaunay


def inpolyhedron(ph, points):
    """
    Given a polyhedron vertices in `ph`, and `points` return 
    critera that each point is either with in or outside the polyhedron
    
    Both polyhedron and points should have the same shape i.e. num_points X num_dimensions
    
    Returns a boolian array : True if inside, False if outside
    
    """
    tri = Delaunay(ph)
    inside = Delaunay.find_simplex(tri, points)
    criteria = inside < 0
    return ~criteria


def is_collinear(grid, tri_coords, simplex):
    """ 
    determines whether a simplex is coplanar when lifted to design space.
    Returns True is a simplex has the lifted coordinates as collinear.
    """
    coords = np.array([tri_coords[x, :] for x in simplex])
    M = np.vstack((coords.T, np.array([1, 1, 1])))
    area = np.linalg.det(M)
    flag = area < 1e-15

    return flag


from math import pi


def get_ternary_coords(point):
    a, b, c = point
    x = 0.5 - a * np.cos(pi / 3) + b / 2;
    y = 0.866 - a * np.sin(pi / 3) - b * (1 / np.tan(pi / 6) / 2);

    return [x, y]


def is_boundary_point(point, zero_value=MIN_POINT_PRECISION):
    if np.isclose(point, MIN_POINT_PRECISION).any():
        return True
    else:
        return False


def is_pure_component(point, zero_value=MIN_POINT_PRECISION):
    counts = Counter(point)
    if counts[MIN_POINT_PRECISION] > 1:
        return True
    else:
        return False


def get_max_delaunay_edge_length(grid):
    delaunay = Delaunay(np.asarray(grid[:-1, :].T))
    max_delaunay_edge = 0.0
    for sx in delaunay.simplices:
        vertex_sx = [grid[:, x] for x in sx]
        edges = combinations(vertex_sx, 2)
        edge_lengths = np.array([norm(e[0] - e[1]) for e in edges])
        current_max = np.max(edge_lengths)
        if max_delaunay_edge < current_max:
            max_delaunay_edge = current_max

    return max_delaunay_edge


""" Main comoutation function """


def serialcompute(configuration, meshsize, **kwargs):
    """
    Main python function to obtain a phase diagram for n-component polymer mixture system.
    
    parameters:
    -----------
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
    flag_lift_label = kwargs.get('flag_lift_label', False)
    use_weighted_delaunay = kwargs.get('use_weighted_delaunay', False)
    lift_grid_size = kwargs.get('lift_grid_size', 200)

    dimensions = len(configuration['M'])

    since = time.time()

    outdict = {}
    thresh_epsilon = 5e-3

    """ Perform a parallel computation of phase diagram """
    # 1. generate grid
    grid = makegridnd(meshsize, dimensions)
    outdict['grid'] = grid

    lap = time.time()
    if verbose:
        print('{}-dimensional grid generated at {:.2f}s'.format(dimensions, lap - since))

    # 2. compute free energy on the grid (parallel)
    CHI = _utri2mat(configuration['chi'], dimensions)
    flag_make_energy_paraboloid = kwargs.get('flag_make_energy_paraboloid', True)
    flag_lift_purecomp_energy = kwargs.get('flag_lift_purecomp_energy', False)

    if np.logical_or(flag_make_energy_paraboloid, flag_lift_purecomp_energy):
        beta = 0.0
    else:
        beta = kwargs.get('beta', 1e-4)
        if verbose:
            print('Using beta (={:.2E}) correction for energy landscape'.format(beta))

    energy = np.asarray([flory_huggins(x, configuration['M'], CHI, beta=beta) for x in grid.T])
    lap = time.time()
    if verbose:
        print('Energy computed at {:.2f}s'.format(lap - since))

    # Make energy a paraboloid like by extending the landscape at the borders
    if flag_make_energy_paraboloid:
        max_energy = np.max(energy)
        pad_energy = kwargs.get('pad_energy', 2)
        if verbose:
            print('Making energy manifold a paraboloid with {:d}x padding of {:.2f} maximum energy'.format(pad_energy,
                                                                                                           max_energy))
        boundary_points = np.asarray([is_boundary_point(x) for x in grid.T])
        energy[boundary_points] = pad_energy * max_energy

    elif flag_lift_purecomp_energy:
        max_energy = np.max(energy)
        pad_energy = kwargs.get('pad_energy', 2)
        if verbose:
            print('Aplpying {:d}x padding of {:.2f} maximum energy to pure components'.format(pad_energy, max_energy))
        pure_points = np.asarray([is_pure_component(x) for x in grid.T])
        energy[pure_points] = pad_energy * max_energy

    outdict['energy'] = energy

    lap = time.time()
    if verbose:
        print('Energy is corrected at {:.2f}s'.format(lap - since))

    # 3. Compute convex hull
    if not use_weighted_delaunay:
        _method = 'Convexhull'
        points = np.concatenate((grid[:-1, :].T, energy.reshape(-1, 1)), axis=1)
        hull = ConvexHull(points)
        outdict['hull'] = hull
    else:
        raise NotImplemented

    lap = time.time()
    if verbose:
        print('{} is computed at {:.2f}s'.format(_method, lap - since))

    # determine threshold
    threshold_type = kwargs.get('threshold_type', 'delaunay')
    if threshold_type == 'delaunay':
        thresh = get_max_delaunay_edge_length(grid) + thresh_epsilon
    elif threshold_type == 'uniform':
        thresh_scale = kwargs.get('thresh_scale', 1.25)
        thresh = thresh_scale * euclidean(grid[:, 0], grid[:, 1])

    if verbose:
        print('Using {:.2E} as a threshold for Laplacian of a simplex'.format(thresh))

    outdict['thresh'] = thresh

    if not flag_refine_simplices:
        simplices = hull.simplices
    else:
        upper_hull = np.asarray([is_upper_hull(grid, simplex) for simplex in hull.simplices])
        simplices = hull.simplices[~upper_hull]

    flag_remove_collinear = kwargs.get('flag_remove_collinear', False)
    if flag_remove_collinear:
        # remove colpanar simplices
        tri_coords = np.array([get_ternary_coords(pt) for pt in grid.T])
        coplanar_simplices = np.asarray([is_collinear(grid, tri_coords, simplex) for simplex in simplices])
        if len(coplanar_simplices) == 0:
            warnings.warn('There are no coplanar simplices.')
        else:
            simplices = simplices[~coplanar_simplices]

        lap = time.time()
        if verbose:
            print('Simplices are refined at {:.2f}s'.format(lap - since))

    outdict['simplices'] = simplices
    if verbose:
        print('Total of {} simplices in the convex hull'.format(len(simplices)))

    # 4. for each simplex in the hull compute number of connected components (parallel)
    num_comps = [label_simplex(grid, simplex, thresh) for simplex in simplices]
    lap = time.time()
    if verbose:
        print('Simplices are labelled at {:.2f}s'.format(lap - since))
    outdict['num_comps'] = num_comps

    if flag_lift_label:

        # 5. lift the labels from simplices to points (parallel)
        if lift_grid_size == meshsize:
            lift_grid = grid
        else:
            lift_grid = makegridnd(lift_grid_size, dimensions)  # we lift labels to a constant mesh

        inside = [lift_label(grid, lift_grid, simplex, label) for simplex, label in zip(simplices, num_comps)]

        flags = [item[1] for item in inside]
        lap = time.time()
        if verbose:
            print('Labels are lifted at {:.2f}s'.format(lap - since))

            print('Total {}/{} coplanar simplices'.format(Counter(flags)[0], len(simplices)))

        phase = np.zeros(lift_grid.shape[1])
        for i, label in zip(inside, num_comps):
            if i[1] == 1:
                phase[i[0]] = label
        phase = phase.reshape(1, -1)
        output = np.vstack((lift_grid, phase))
        index = ['Phi_' + str(i) for i in range(1, output.shape[0])]
        index.append('label')
        output = pd.DataFrame(data=output, index=index)

    else:
        output = []

    outdict['output'] = output

    lap = time.time()
    print('Computation took {:.2f}s'.format(lap - since))

    return outdict
