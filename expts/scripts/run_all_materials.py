"""
Parameters are taken from the paper : 

"""
import numpy as np
import pdb

import sys
if '../' not in sys.path:
    sys.path.insert(0,'../')
    sys.path.insert(0,'./')

from solvers.utils import makegrid3d, scatter3d,contours
from solvers import spinodal, cem, helpers
from plotly.subplots import make_subplots

import mpltern
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# set up a constant grid for 3D
grid = np.asarray(makegrid3d(num=40))

# setup material parameters
floryhuggins =[{'name': 'CHCl3/F8/PCBM','chi': [0.341,0.885,0.941]},
              {'name': 'chlorobenzene/PFB/PCBM','chi': [0.340,0.899,0.899]},
              {'name': 'CHCl3/APFO-3/PCBM','chi': [0.505,0.885,0.450]},
              {'name': 'chlorobenzene/APFO-3/PCBM','chi': [0.480,0.899,0.479]}, 
              {'name': 'xylene/PFB/PCBM','chi': [0.442,1.630,1.013]}
               ]
Nset = {'CHCl3':0.1,'chlorobenzene':0.1,'xylene':0.1,'F8':720,'PFB':73,'F8BT':172,'APFO-3':66,'PCBM':5}

mplfig,axs = plt.subplots(len(floryhuggins),1,figsize=(10.8, 19.2),subplot_kw={'projection':'ternary','rotation':-120})
mplfig.subplots_adjust(wspace=0.65,hspace = 0.45)
plt.set_cmap('bwr')

from solvers.utils import set_ternlabel

def plot_phase_diagram(curvature,score,ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10.8, 4.8))
        fig.subplots_adjust(left=0.075, right=0.85, wspace=0.65)
        ax = fig.add_subplot(1,1,1,projection='ternary',rotation=-120)
    colors = ["faded green", "dusty purple","amber", "greyish"]
    colors = sns.xkcd_palette(colors)
    t,l,r = grid[:,0],grid[:,1],grid[:,2]

    stable_ids = np.where(np.asarray(curvature)==2)
    binodal_ids = np.where(np.asarray(score)>1.75)
    refined_binodal_ids = []
    for ind in binodal_ids[0]:
        is_binary = np.any(np.round(grid[ind],2)<=0.03)
        if not is_binary:
            refined_binodal_ids.append(ind)

    metastable_ids = np.setdiff1d(refined_binodal_ids,stable_ids)
    ax.scatter(t[stable_ids], l[stable_ids], r[stable_ids],c=colors[0],label='stable',s=15)
    ax.scatter(t[refined_binodal_ids], l[refined_binodal_ids], r[refined_binodal_ids],c=colors[1],label='meta-stable',s=15)
    ax = set_ternlabel(ax)
    
    return ax

for i,entry in enumerate(floryhuggins):
    chi = np.asarray(entry['chi'])+3
    material = entry['name'].split("/")
    solvent,polymer,PCBM = material
    M = [Nset[solvent],Nset[polymer],Nset[PCBM]]
    
    CHI = np.array([[0,chi[0],chi[1]],[chi[0],0,chi[2]],[chi[1],chi[2],0]])
 
    print('Calculating: {}'.format(material))

    gmix = lambda x: helpers.flory_huggins(x, M, CHI, beta=1e-4)

    energy = []
    for point in grid:
        energy.append(gmix(point))

    sp = spinodal.Spinodal(grid,gmix,energy=energy)
    bp = cem.CEM(grid,gmix,energy=energy)
    coords = bp.coords   

    curvature = sp.get_spinodal()
    score = bp.get_binodal()

    ax = axs[i]
    ax = plot_phase_diagram(curvature,score,ax=ax)
    ax.set_title( ('-').join(material),pad=40)
    ax.legend(loc='upper left')
    
plt.tight_layout()   
plt.savefig('./data/paper_materials.pdf',dpi=500,bbox_inches='tight')
        
        
        