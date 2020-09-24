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

from polyphase import serialcompute, timer

from numpy.linalg import norm
from scipy.constants import gas_constant
from itertools import combinations, product

import os
dirname = '../figures/hteplotsV2/'
if not os.path.exists(dirname):
    os.makedirs(dirname)

import ray

""" The following is required for multinode parallelization """
if os.environ.get("ip_head") is not None:
    ray.init(address='auto', node_ip_address=os.environ["ip_head"].split(":")[0],redis_password=os.environ["redis_password"])
    num_nodes = len(ray.nodes())
    print('Total number of nodes are {}'.format(num_nodes))
    NUM_CPUS=5
else:
    ray.init(local_mode=False)
    print('Using single node all core paralleization')
    NUM_CPUS=1


hte_df = pd.read_csv('../data/htev2.csv')
    
        
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


@ray.remote(num_cpus=NUM_CPUS)
def plot_phase_diagram(row):
    
    chi = [row['chi12'], row['chi13'], row['chi23']]
    
    fname = dirname +'{}_{}_{}'.format(row['solvent'], row['SM'], row['polymer'])
    remote_id = ray.services.get_node_ip_address()
    print('Computing {} on {}'.format(fname, remote_id))
    
    M = row['dop']
    configuration = {'M': M, 'chi':chi}
    dx = 400
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
        'thresh_scale':0.1*dx,
        'lift_grid_size':dx,
        'verbose' : False
     }

    out = serialcompute(configuration, dx, **kwargs)
    output = out['output']

    plain_phase_diagram(output)
    plt.savefig(fname,dpi=500, bbox_inches='tight')
    plt.close()
    
    del out, output, chi, M, configuration, dx, kwargs
    
    return fname

T = timer()
PM6_Y6 = hte_df.loc[(hte_df['SM'] == 'Y6') & (hte_df['polymer'] == 'PM6')]

remaining_result_ids  = [plot_phase_diagram.remote(i) for _,i in PM6_Y6.iterrows()]

while len(remaining_result_ids) > 0:
    ready_result_ids, remaining_result_ids = ray.wait(remaining_result_ids, num_returns=1)
    result_id = ready_result_ids[0]
    result = ray.get(result_id)
    print('Processed : {}'.format(result)) 


print('Program took {}'.format(timer.end()))    
    
    