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
from math import pi

def touchup3d(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    ax.zaxis._axinfo['juggled'] = (1,2,0)

def get_ternary_coords(point):
    """ Compute 2d embedding of a 3d hyperplane """
    a,b,c = point
    x = 0.5-a*np.cos(pi/3)+b/2;
    y = 0.866-a*np.sin(pi/3)-b*(1/np.tan(pi/6)/2);
    
    return [x,y]
    
def get_simplex_area(engine, simplex_id):
    matrix = []
    for s in engine.simplices[simplex_id]:
        v = engine.grid[:,s]
        matrix.append(get_ternary_coords(v))

    matrix = np.hstack((np.asarray(matrix), np.ones((3,1))))
    area = 0.5*np.linalg.det(matrix)
    return np.abs(area)

def get_phase_area(engine, phase_label):
    total_area = 0
    phase_simplices = np.where(np.asarray(engine.num_comps)==phase_label)[0]
    if len(phase_simplices)==0:
        return 0
    for ps in phase_simplices:
        total_area += get_simplex_area(engine, ps)
    
    return total_area/0.43   

Mp = 10
chi12 = np.linspace(1.3,1.44, num=5)
chi13 = [0.3,0.6,1,2,3]
chi23 = [0.3,0.6,1,2,3]
chispace = list(product(chi12, chi13, chi23))
chispace = np.asarray(chispace)
df = pd.DataFrame(chispace, columns=['chi12', 'chi13', 'chi23'])
df.to_pickle("../figures/chispace/{}.pkl".format(Mp))

dirname = '../figures/chispace/chinamed/'
if os.path.exists(dirname):
    shutil.rmtree(dirname)    
os.makedirs(dirname)

num_simplices = []
for i, chi in df.iterrows():
    M = [Mp,10,1]
    f = lambda x: polyphase.flory_huggins(x , M, chi)
    engine = polyphase.PHASE(f, 200,3)
    engine.compute(use_parallel=False)
    num_simplices.append([get_phase_area(engine, 1), get_phase_area(engine, 2),
                          get_phase_area(engine, 3)])
    
    polyphase.plain_phase_diagram(engine.df)
    fname = '_'.join('{}'.format(i) for i in chi).replace('.','p')
    plt.savefig(dirname+'{}.png'.format(fname), bbox_inches='tight', dpi=300)
    plt.close()
    
fig, axs = plt.subplots(1,3, figsize=(3*6,6),subplot_kw={'projection':'3d'})
fig.subplots_adjust(wspace=0.4)
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

