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
  
# set up a constant grid for 3D
grid = np.asarray(makegrid3d(num=40))

# set up parameters 
chi = [0.442,1.630,1.013]
CHI = np.array([[0,chi[0],chi[1]],[chi[0],0,chi[2]],[chi[1],chi[2],0]])

# Loop over parameters
Ntrials = [64]
Betatrials = [1e-3,2e-3,4e-3,6e-3,8e-3,10e-3]
rolcols = [[1,1],[1,2],[2,1],[2,2],[3,1],[3,2]]


for i,N in enumerate(Ntrials):
    fig = make_subplots(rows=3, cols=2, \
                        specs=[[{'type': 'scene'}, {'type': 'scene'}],[{'type': 'scene'}, {'type': 'scene'}],\
                               [{'type': 'scene'}, {'type': 'scene'}]],\
                        subplot_titles=("beta=1e-3", "beta=2e-3", "beta=4e-3", "beta=6e-3","beta=8e-3","beta=1e-2"))
    
    mplfig,axs = plt.subplots(len(Betatrials),2,figsize=(10.8, 19.2),subplot_kw={'projection':'ternary','rotation':-120})
    mplfig.subplots_adjust(wspace=0.65,hspace = 0.45)
    plt.set_cmap('bwr')
    
    for mplind,beta in enumerate(Betatrials):
        M = np.array([1.0,N,5])
        print('Calculating: N={},beta={}'.format(N,beta))
        
        gmix = lambda x: helpers.flory_huggins(x, M, CHI, beta=beta)
        
        energy = []
        for point in grid:
            energy.append(gmix(point))
            
        sp = spinodal.Spinodal(grid,gmix,energy=energy)
        bp= cem.CEM(grid,gmix,energy=energy)
        coords = bp.coords   
        
        trace = scatter3d(coords,energy)
        r, c = rolcols[mplind]
        fig.add_trace(trace, row=r, col=c)
        fig.update_layout(scene = dict(zaxis_title='Energy'))
        
        
        curvature = sp.get_spinodal()
        score = bp.get_binodal()
        
        ax = axs[mplind,0]
        ax,cs,cax = contours(grid,score,mode='contour',ax=ax,level=[0.99])
        pad_title = 24
        colorbar = mplfig.colorbar(cs, cax=cax)
        colorbar.set_label('CEM Score beta='+str(beta), rotation=270, va='baseline')

        ax = axs[mplind,1]

        ax,cs,cax = contours(grid,curvature,mode='contour',ax=ax,level=[1.00])
        pad_title = 36
        colorbar = mplfig.colorbar(cs, cax=cax)
        colorbar.set_label('Spinodal N='+str(beta), rotation=270, va='baseline')

    fig.update_layout(title_text='N='+str(N),height=1200,width=800)    
    fig.write_html('./data/FinerBeta_N_' + str(N) +'.html')

    plt.savefig('./data/FinerBetaPhasediagrams_N' +str(N)+'.pdf',dpi=500,bbox_inches='tight')
        
        
        