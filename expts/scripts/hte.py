import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import mpltern

from polyphase import (PHASE, timer, 
                       flory_huggins, TestPhaseSplits)
from itertools import product

import os
import shutil
from matplotlib.cm import ScalarMappable
from matplotlib import colors
plt.rcParams.update({'font.size': 22})

dirname = '../figures/artificial/'
if os.path.exists(dirname):
    shutil.rmtree(dirname)    
os.makedirs(dirname)
   
import ray
num_nodes = float(os.environ.get('SLURM_JOB_NUM_NODES'))
""" The following is required for multinode parallelization """
if num_nodes>1:
    ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0],
             _redis_password=os.environ["redis_password"])
    num_nodes = len(ray.nodes())
    print('Total number of nodes are {}'.format(num_nodes))
    NUM_CPUS=5
elif num_nodes==1:
    ray.init(local_mode=True)
    print('Using single node all core paralleization')
    NUM_CPUS=1
else:
    quit()

# set up common data
hte_df = pd.read_pickle('../data/artificial_solvents.pkl')


def plain_phase_diagram(output, ax = None):
    """ 
    Plot phase diagrams as points without any labels or stuff
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection':'ternary'})
    else:
        fig = plt.gcf()
    
    phase_colors =['w','r','g','b']
    df = output.transpose()
    for i, p in df.groupby('label'):
        ax.scatter(p['Phi_3'], p['Phi_1'], p['Phi_2'], c=phase_colors[int(i)])
        
    plt.axis('off')
    
    return ax    

@ray.remote(num_cpus=NUM_CPUS)
def plot_phase_diagram(row):
    fname = dirname +'{}_{}_{}.png'.format(int(row['solvent']), row['SM'], row['polymer'])
    remote_id = ray.services.get_node_ip_address()
    print('Computing {} on {}'.format(fname, remote_id))
    
    M = row['dop']
    chi = [row['chi12'], row['chi13'], row['chi23']]
    dx = 400
    
    f = lambda x : flory_huggins(x, M, chi)
    engine = PHASE(f,dx, len(M))
    
    engine.compute()
    flag, fail, total = raise_flag(engine)
    
    if not flag:
        describe_twophase_split_tests(engine)    
        plt.savefig(dirname+'test_{}.png'.format(int(row['solvent'])), dpi=500)
        plt.close()
        
    plain_phase_diagram(engine.df)
    fname = dirname +'{}_{}_{}_{}_{}_{}.png'.format(flag,fail,total,
        int(row['solvent']),row['SM'], row['polymer'])

    plt.savefig(fname,dpi=500, bbox_inches='tight')
    plt.close()
    
    del engine, f

    return fname, flag

def raise_flag(engine):
    results = []
    for PHASE_ID in [2,3]:
        phase_simplices_ids = np.where(np.asarray(engine.num_comps)==PHASE_ID)[0]
        for simplex_id in phase_simplices_ids:
            test = TestPhaseSplits(engine,phase=PHASE_ID,
                                             simplex_id=simplex_id, threshold=0.05)
            results.append(test.check_centroid())
    
    results = np.asarray(results)
    
    failed = np.sum(~results)
    total = len(results)
    ratio = failed/total
    decision = ratio<0.1
    
    return decision, failed, total

def describe_twophase_split_tests(engine):
    PHASE_ID = 2
    phase_simplices_ids = np.where(np.asarray(engine.num_comps)==PHASE_ID)[0]
    failed, min_splits = [],[]
    fig = plt.figure(constrained_layout=True, figsize=(6*4,4*4))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    phase_colors =['w','r','g','b']
    cmap = colors.ListedColormap(phase_colors[1:])
    df = engine.df.transpose()
    for i, p in df.groupby('label'):
        ax1.scatter(p['Phi_1'], p['Phi_2'], c=phase_colors[int(i)])    
    boundaries = np.linspace(1,4,4)
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable,shrink=0.5, aspect=5, ticks=[1.5,2.5,3.5],ax=ax1)
    cbar.ax.set_yticklabels(['1-Phase', '2-Phase', '3-Phase'])    
    ax1.set_xlabel(r'$\phi_1$')
    ax1.set_ylabel(r'$\phi_2$')

    criteria = engine.df.T['label']==PHASE_ID
    df = engine.df.T[criteria]
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(df['Phi_1'], df['Phi_2'], color='k')
    ax2.set_xlabel(r'$\phi_1$')
    ax2.set_ylabel(r'$\phi_2$')
    ax2.set_title('2-phase region')

    ax3 = fig.add_subplot(gs[1, 0])
    for simplex_id in phase_simplices_ids:
        test = TestPhaseSplits(engine,phase=PHASE_ID,simplex_id=simplex_id, threshold=0.05)
        decision = test.check_centroid()
        if not decision:
            failed.append([simplex_id, test.centroid_splits_])
            test.visualize_centroid(ax=ax3, show=False)
        min_splits.append(min(test.centroid_splits_))
    ax3.set_xlabel(r'$\phi_1$')
    ax3.set_ylabel(r'$\phi_2$')  

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(min_splits, density=False)
    ax4.axvline(x=0.05, ls='--', color='k')
    ax4.set_xlabel('Minimum split ratio')
    ax4.set_ylabel('Counts')  
    fig.suptitle('{}/{} simplices failed the test'.format(len(failed), len(phase_simplices_ids)))
    
    return

T = timer()
remaining_result_ids  = [plot_phase_diagram.remote(i) for _,i in hte_df.iterrows()]
num_total = len(hte_df)
num_failed = 0
while len(remaining_result_ids) > 0:
    ready_result_ids, remaining_result_ids = ray.wait(remaining_result_ids, num_returns=1)
    result_id = ready_result_ids[0]
    result, flag = ray.get(result_id)
    if not flag:
        num_failed += 1
    print('Processed : {}'.format(result)) 

print('Total of {}/{} failed'.format(num_failed,num_total))
print('Program took {}'.format(T.end()))    
    
    