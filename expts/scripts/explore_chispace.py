import numpy as np
import pandas as pd
import polyphase
from itertools import product
import polyphase
import matplotlib.pyplot as plt
import os, shutil
from collections import Counter
plt.rcParams.update({
    "text.usetex": True})

def touchup3d(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    ax.zaxis._axinfo['juggled'] = (1,2,0)

Mp = 10
chi12 = np.linspace(1.3,1.44, num=5)
chi13 = [0.3,0.6,1,2,3]
chi23 = [0.3,0.6,1,2,3]
chispace = list(product(chi12, chi13, chi23))
chispace = np.asarray(chispace)
df = pd.DataFrame(chispace, columns=['chi12', 'chi13', 'chi23'])

dirname = '../figures/chispace/dimred/'
if os.path.exists(dirname):
    shutil.rmtree(dirname)    
os.makedirs(dirname)

num_simplices = []
for i, chi in df.iterrows():
    M = [Mp,10,1]
    f = lambda x: polyphase.flory_huggins(x , M, chi, logapprox=True)
    engine = polyphase.PHASE(f, 200,3)
    engine.compute(use_parallel=False)
    phaselabels = engine.num_comps 
    twophase = np.sum(np.asarray(phaselabels)==2)
    onephase = np.sum(np.asarray(phaselabels)==1)
    threephase = np.sum(np.asarray(phaselabels)==3)
    num_simplices.append([onephase, twophase, threephase])
    
    polyphase.plain_phase_diagram(engine.df)
    plt.savefig(dirname+'{}.png'.format(i), bbox_inches='tight', dpi=300)
    plt.close()
    
fig, axs = plt.subplots(1,3, figsize=(3*6,6),subplot_kw={'projection':'3d'})
cvalues = np.asarray(num_simplices)
x = df.to_numpy()


for i,ax in enumerate(axs):
    path = ax.scatter(x[:,0], x[:,1], x[:,2],c=cvalues[:,i],
                      alpha=1.0, cmap='bwr')
    fig.colorbar(path, pad=0.05, ax = ax, fraction=0.035)
    
    ax.set_xlabel(r'$\chi_{12}$')
    ax.set_ylabel(r'$\chi_{13}$')
    ax.set_zlabel(r'$\chi_{23}$')
    ax.set_title('{}-phase'.format(i+1))
    touchup3d(ax)
    
plt.savefig('../figures/chispace/{}.png'.format(Mp), dpi=500, bbox_inches='tight')

