import numpy as np
import pdb

# import sys
# from os import path
# filepath = path.dirname( path.dirname(path.abspath(__file__)))
# if filepath not in sys.path:
#     sys.path.append(filepath)
    
import sys
if '../' not in sys.path:
    sys.path.insert(0,'../')
    
from parallel.parphase import compute
from solvers.visuals import plot_mpltern, plot_lifted_label_ternary
from solvers.helpers import flory_huggins, utri2mat

import mpltern
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
plt.close('all')

from solvers.utils import compute_chemical_potential
from itertools import combinations, product
from scipy.spatial.distance import cosine
from autograd import grad, jacobian, hessian
from numpy.linalg import norm

dirname = '../figures/convergence/'

test_config_list = [{'M':np.array([64,1,1]), 'chi': [1.0,0.3,0.2] }, # 0 : B Zhou Page 65
                    {'M':np.array([5,5,1]), 'chi': [1.0,0.5,0.5]}, # 1: oW APS
                    {'M':np.array([20,20,1]), 'chi': [1.0,0.5,0.5]}, # 2: 
                    {'M':np.array([20,5,1]), 'chi': [0.5,0.34,0.534]}, # 3
                    {'M':np.array([60,5,1]), 'chi': [0.5,0.34,0.534]}  # 4   
              ]

test_config_id= 1

dimensions = 3
configuration = test_config_list[test_config_id]
name_prefix = '{}_'.format(test_config_id)   
    
"""
test 1: Listing labels to multiple uniform grids

We first compute a simplex labelling for different mesh sizes : 100, 200, 400
next we lift these labels to different uniform grids :100, 200, 400
"""

def test1():
    mesh_grids = np.array([100,200,400])
    lift_grids = np.array([100,200,400])
    fig, axs = plt.subplots(3,3,figsize=(27,27),subplot_kw=dict(projection='ternary'))
    fig.subplots_adjust(hspace=0.3, wspace =0.4)
    for i, m in enumerate(mesh_grids):
        for j, l in enumerate(lift_grids):
            print('Grid size : {}, Lift grid : {}'.format(m, l))
            outdict = compute(dimensions, configuration, m,
                                             flag_refine_simplices=True, flag_lift_label=True, lift_grid_size=l)
            output, simplices, grid, num_comps = outdict['output'], outdict['simplices'], outdict['grid'], outdict['num_comps']
            ax, cbar = plot_lifted_label_ternary(output, ax = axs[i,j])
            ax.set_title('Mesh size: {} Lifted to : {}'.format(m,l), pad=15)
            
            del output, simplices, grid, num_comps, outdict
            
    plt.savefig(dirname + name_prefix + 'test1.png',dpi=500, bbox_inches='tight')
    plt.close()

    
"""
test 2: Sweep over two hyper parameters 
Use thresholds of [5x, 10x, 20x] or [1x,2x,5x] and mesh sizes of [100,200,400]
"""

def test2():
    mesh_grids = np.array([100,200,300, 400])
    fig, axs = plt.subplots(4,3,figsize=(4*3*1.6,4*4),subplot_kw=dict(projection='ternary'))
    fig.subplots_adjust(hspace=0.3, wspace =0.4)
    for i, m in enumerate(mesh_grids):
        thresholds = np.array([0.05,0.1,0.2])*m
        for j, t in enumerate(thresholds):
            kwargs = {
                'flag_refine_simplices':True,
                'flag_lift_label': False,
                'use_weighted_delaunay': False,
                'flag_remove_collinear' : False, 
                'beta':1e-4, # not used 
                'flag_make_energy_paraboloid': True, 
                'pad_energy': 2,
                'flag_lift_purecomp_energy': False,
                'threshold_type':'uniform',
                'thresh_scale':t 
             }
            
            print('Grid size : {}, Threshold: {}'.format(m, t))
            outdict = compute(dimensions, configuration, m, **kwargs)
            output, simplices, grid, num_comps = outdict['output'], outdict['simplices'], outdict['grid'], outdict['num_comps']
            ax, cbar = plot_mpltern(grid, simplices, num_comps, ax = axs[i,j])
            ax.set_title('Mesh size: {} Threshold : {}'.format(m,t), pad=15, fontsize=20)
            
            del output, simplices, grid, num_comps, outdict
            
    plt.savefig(dirname + name_prefix + 'test2.png',dpi=500, bbox_inches='tight')
    plt.close()
            
"""
test3: Figure out how far off the chemical potentials are

for different mesh sizes, figureout how far off a threshold chemical potentials of simplices are
"""

# define a helper to return a counts of numpy array.
# This helper is used in tests 3, 4

def _get_counts(grid, simplices, num_comps, function):
    counts = np.zeros((3,4))
    for simplex, label in zip(simplices,num_comps):
        mu = np.array([function(x) for x in simplex])

        comb = combinations(mu, 2)
        dists = np.array([cosine(x[0], x[1]) for x in comb])

        if np.all(dists<0.05):
            counts[label-1,0] += 1
        elif np.all(dists<0.1):
            counts[label-1,1] += 1
        elif np.all(dists<0.2):
            counts[label-1,2] += 1
        else:
            counts[label-1,3] += 1

    return counts


def test3():
    mesh_grids = np.array([100,200,400])
    counts = np.zeros((3,4,3))
    for i, m in enumerate(mesh_grids):
        outdict = compute(dimensions, configuration, m,
                                         flag_refine_simplices=True, flag_lift_label=False)
        output, simplices, grid, num_comps = outdict['output'], outdict['simplices'], outdict['grid'], outdict['num_comps']
        
        function = lambda x: compute_chemical_potential(grid[:,x],configuration['M'],configuration['chi'])
        counts[:,:,i] = _get_counts(grid, simplices,num_comps, function)
                
        del output, simplices, grid, num_comps, outdict   
        
    np.save(dirname+ name_prefix +'test3.npy', counts)
                

"""
Test 4 : Figure out if a phase is obtained between points that have a common tangent.
We first compute a normal vector for the energy manifold given as M(e,phi1,phi2,...).
A normal vector would be : [dM/d(phi_1), dM/d(phi_2), dM/d(phi_3), ... ]
We then compute angles between normal vector for vertices of simplices and sort them into three category as done in test 3.

"""

def test4():
    mesh_grids = np.array([100,200,400])
    counts = np.zeros((3,4,3))
    
    gmix = lambda x: flory_huggins(x, configuration['M'], utri2mat(configuration['chi'], dimensions),beta=1e-4)
    normal = jacobian(gmix) 
    for i, m in enumerate(mesh_grids):
        print('Grid size : {}'.format(m))
        outdict = compute(dimensions, configuration, m,
                                         flag_refine_simplices=True, flag_lift_label=False)
        
        output, simplices, grid, num_comps = outdict['output'], outdict['simplices'], outdict['grid'], outdict['num_comps']
        
        function = lambda x: normal(grid[:,x])
        counts[:,:,i] = _get_counts(grid, simplices,num_comps, function)
      
        del output, simplices, grid, num_comps, outdict   
        
    np.save(dirname+ name_prefix +'test4.npy', counts)        

    
"""
Test 5: Change beta parameter and plot all the phase diagrams
"""
def test5():
    mesh_grids = np.array([100,200,400])
    betas = np.array([1e-4,5e-4,1e-5])
    fig, axs = plt.subplots(3,3,figsize=(27,27),subplot_kw=dict(projection='ternary'))
    fig.subplots_adjust(hspace=0.2, wspace =0.3)
    for i, m in enumerate(mesh_grids):
        for j, b in enumerate(betas):
            print('Grid size : {}, Beta: {:.2E}'.format(m, b))
            outdict = compute(dimensions, configuration, m,
                                             flag_refine_simplices=True, flag_lift_label=True, beta=b)
            output, simplices, grid, num_comps = outdict['output'], outdict['simplices'], outdict['grid'], outdict['num_comps']
            ax, cbar = plot_mpltern(grid, simplices, num_comps, ax = axs[i,j])
            ax.set_title('Mesh size: {} Beta : {:.2E}'.format(m,b), pad=15)
            
            del output, simplices, grid, num_comps, outdict
            
    plt.savefig(dirname + name_prefix + 'test5.png',dpi=300, bbox_inches='tight')
    plt.close()    

"""
Test 6 : Check if normals at the vertices match normal to the plane
Normal at the vertices are computed similar to in test 3, 4 i.e. 
a normal vector for the energy manifold given as M(e,phi1,phi2,...)is n =  [dM/d(phi_1), dM/d(phi_2), dM/d(phi_3), ... ]

Given three points on the plane as P, Q, R one can compute normal to the plane using:
n_p = PQ x PR where XY represents vector passing from X to Y and 'x' is vector cross product (or more techincally a tensor product)

Now we compare each of the three normal vectors in `n` to `n_p`
"""

def test6():
    mesh_grids = np.array([100,200,400])
    counts = np.zeros((3,4,3)) # in the order of  labels, type, mesh_grid
    
    gmix = lambda x: flory_huggins(x, configuration['M'], utri2mat(configuration['chi'], dimensions),beta=1e-4)
    normal = jacobian(gmix) 
    for i, m in enumerate(mesh_grids):
        print('Grid size : {}'.format(m))
        outdict = compute(dimensions, configuration, m,
                                         flag_refine_simplices=True, flag_lift_label=False)
        
        output, simplices, grid, num_comps = outdict['output'], outdict['simplices'], outdict['grid'], outdict['num_comps']
        
        
        for simplex, label in zip(simplices,num_comps):
            n_vertices = np.array([normal(grid[:,x]) for x in simplex])
            verticies = np.asarray([grid[:,x] for x in simplex])
            PQ = verticies[0,:] - verticies[1,:]
            QR = verticies[1,:] - verticies[2,:]
            n_plane = -1*np.cross(PQ, QR).reshape(1,3)
            set_product = product(*[n_vertices, n_plane])                
            dists = np.array([cosine(x[0], x[1]) for x in set_product])
            if np.all(dists<0.05):
                counts[label-1,0,i] += 1
            elif np.all(dists<0.1):
                counts[label-1,1,i] += 1
            elif np.all(dists<0.2):
                counts[label-1,2,i] += 1
            else:
                counts[label-1,3,i] += 1

        del output, simplices, grid, num_comps, outdict   
        
    np.save(dirname+ name_prefix +'test6.npy', counts) 


"""
Test 7 : We test our hypothesis that a threshld of = 0.1*mesh_size is a sweet spot for the hyperparameter.
This test involves computing phase diagrams for multiple mesh sizes and mapping simplex labels to points on corresponding uniform grid.
"""
def test7():
    mesh_grids = np.array([25,50,300])
    lift_grids = mesh_grids
    thresholds = 0.1*mesh_grids
    fig, axs = plt.subplots(3,2,figsize=(2*3*1.6,3*3),subplot_kw=dict(projection='ternary'))
    fig.subplots_adjust(hspace=0.4, wspace =0.3)
    for i, m in enumerate(mesh_grids):
        l = lift_grids[i]
        t = thresholds[i]
        print('Grid size : {}, Lift grid : {}'.format(m, l))
        outdict = compute(dimensions, configuration, m, thresh=t,
                                         flag_refine_simplices=True, flag_lift_label=True, lift_grid_size=l)
        output, simplices, grid, num_comps = outdict['output'], outdict['simplices'], outdict['grid'], outdict['num_comps']
        ax, cbar = plot_lifted_label_ternary(output, ax = axs[i,0])
        ax.set_title('Mesh size: {} Lifted to : {}'.format(m,l), pad=15)
        
        ax, cbar = plot_mpltern(grid, simplices, num_comps, ax = axs[i,1])
        ax.set_title('Mesh size: {} Threshold : {:.2f}'.format(m,outdict['thresh']), pad=15)
            
    plt.savefig(dirname + name_prefix + 'test7.png',dpi=300, bbox_inches='tight')
    plt.close()    


"""
Test 8:
This test performs a sweep over mesh size and computes phase diagram with threhsold as maximal delaunay edge length.

"""
def test8():
    mesh_grids = np.array([100,200,300, 400])
    fig, axs = plt.subplots(2,2,figsize=(2*4*1.6,4*2),subplot_kw=dict(projection='ternary'))
    axs = axs.flatten()
    fig.subplots_adjust(hspace=0.3, wspace =0.2)
    kwargs = {
        'flag_refine_simplices':True,
        'flag_lift_label': False,
        'use_weighted_delaunay': False,
        'flag_remove_collinear' : False, 
        'beta':1e-4, # not used 
        'flag_make_energy_paraboloid': True, 
        'pad_energy': 2,
        'flag_lift_purecomp_energy': False,
        'threshold_type':'delaunay',
        'thresh_scale':1 # not used
             }
    for i, m in enumerate(mesh_grids):
        print('Mesh size : {}'.format(m))
        outdict = compute(dimensions, configuration, m, **kwargs)
        output, simplices, grid, num_comps = outdict['output'], outdict['simplices'], outdict['grid'], outdict['num_comps']
        ax, cbar = plot_mpltern(grid, simplices, num_comps, ax = axs[i])
        ax.set_title('Mesh size: {}'.format(m), pad=15, fontsize=20)
        
        del output, simplices, grid, num_comps, outdict
            
    plt.savefig(dirname + name_prefix + 'test8.png',dpi=500, bbox_inches='tight')
    plt.close()    
        
if __name__ == '__main__':
#     print('performing test 1')
#     test1()
    
    print('performing test 2')
    test2()
    
#     print('performing test 3')
#     test3()

#     print('performing test 4')
#     test4()
    
#     print('performing test 5')
#     test5()

#     print('performing test 6')
#     test6()

#     print('performing test 7')
#     test7()

    print('performing test 8')
    test8()

    print("Program ended")












