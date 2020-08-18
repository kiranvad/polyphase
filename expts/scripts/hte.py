import numpy as np
import pandas as pd
import pdb

from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt
import mpltern

import sys
if '../' not in sys.path:
    sys.path.insert(0,'../')

import polyphase as phase

from numpy.linalg import norm
from scipy.constants import gas_constant
from itertools import combinations, product

import os
dirname = '../figures/hteplots/'
if not os.path.exists(dirname):
    os.makedirs(dirname)

import ray

# ip_head and redis_passwords are set by ray cluster shell scripts
print(os.environ["ip_head"], os.environ["redis_password"])
ray.init(address='auto', node_ip_address=os.environ["ip_head"].split(":")[0],redis_password=os.environ["redis_password"])

print("Number of Nodes in the Ray cluster: {}".format(len(ray.nodes())))

print('Dashboard URL: http://{}'.format(ray.get_webui_url()))

import pickle
# Load your data from dictonary to pandas data frame
with open('./data/htpdata/solubility.pkl', 'rb') as handle:
    data = pickle.load(handle)
    
axes = [np.arange(0,len(data['solvents'])),np.arange(0,len(data['small molecules'])),np.arange(0,len(data['polymers']))]

data_ray = ray.put(data)
    
import time

def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    
def compute_chi(delta_i,delta_j,V):
    """
    total solubility parameters delta_i, delta_j are computed from hydrogen, polar, dispersive components
    
    delta_i, delta_j in MPa^{1/2} and V in cm3/mol
    
    returns a scalar chi value
    
    """
    constant = 1.0 #4.184*(2.045**2)/(8.314)
    
    chi_ij =  0.34+(constant)*(V/(gas_constant*300)*(delta_i - delta_j)**2)
        
    return chi_ij

def compute_weighted_chi(vec1,vec2,V, W):
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
        chi = [compute_chi(i[0],i[1],V0) for i in combs]
    elif approach==2:
        chi = [compute_weighted_chi(i[0],i[1],V0, W = [1.0,1.0,1.0]) for i in combs] 
    elif approach==3:
        chi = [compute_weighted_chi(i[0],i[1],V0, W = [1.0,0.25,0.25]) for i in combs]            
    else:
        raise KeyError
        
    return chi, inds
        
    
def plain_phase_diagram(output, ax = None):
    """ 
    Plot phase diagrams as points without any labels or stuff
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection':'ternary'})
    else:
        fig = plt.gcf()
    
    phase_colors =['r','g','b']
    df = output.transpose()
    for i, p in df.groupby('label'):
        ax.scatter(p['Phi_3'], p['Phi_1'], p['Phi_2'], c=phase_colors[int(i-1)])
        
    plt.axis('off')
    
    return ax    


@ray.remote
def plot_phase_diagram(data,point):
    
    delta_solvent = data['solvents'].loc[point[0]].tolist()
    delta_sm = data['small molecules'].loc[point[1]].tolist()[2:5]
    delta_polymer = data['polymers'].loc[point[2]].tolist()[2:5]
    fname = dirname +'{}_{}_{}'.format(point[0],data['small molecules']['name'].loc[point[1]],\
                            data['polymers']['name'].loc[point[2]])
    
    chi = get_chi_vector([delta_polymer,delta_sm,delta_solvent], 100, 2)[0]

    remote_id = ray.services.get_node_ip_address()
    print('Computing {} on {}'.format(fname, remote_id))
    
    M = [100,5,1]
    configuration = {'M': M, 'chi':chi}
    dx = 200
    kwargs = {
        'flag_refine_simplices':True,
        'flag_lift_label': True,
        'use_weighted_delaunay': False,
        'flag_remove_collinear' : False, 
        'beta':1e-4, # not used 
        'flag_make_energy_paraboloid': True, 
        'pad_energy': 2,
        'flag_lift_purecomp_energy': False,
        'threshold_type':'uniform',
        'thresh_scale':0.1*dx 
     }

    out = phase.serialcompute(len(M), configuration, dx, **kwargs)
    grid = out['grid']
    num_comps = out['num_comps']
    simplices = out['simplices']
    output = out['output']

    plain_phase_diagram(output)
    plt.savefig(fname,dpi=500, bbox_inches='tight')
    plt.close()
    
    del out, output, chi, M, configuration, dx, kwargs
    del delta_solvent, delta_sm, delta_polymer
    del grid, num_comps, simplices
    
    return fname

start = time.time()

remaining_result_ids  = [plot_phase_diagram.remote(data_ray, i) for i in product(*axes)]

while len(remaining_result_ids) > 0:
    ready_result_ids, remaining_result_ids = ray.wait(remaining_result_ids, num_returns=1)
    result_id = ready_result_ids[0]
    result = ray.get(result_id)
    print('Processed : {}'.format(result)) 


end = time.time()
del data_ray

print('Program took {} seconds'.format(timer(start, end)))    
    
    