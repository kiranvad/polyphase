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

        
NUM_PER_DIMENSION = 5
RUN_IDENTIFIER = 'IDIC_PTB7-Th'

delta_d = np.linspace(15,20, num=NUM_PER_DIMENSION)
delta_p = np.linspace(1,10, num=NUM_PER_DIMENSION)
delta_h = np.linspace(1,10, num=NUM_PER_DIMENSION)
solvents = list(product(delta_d, delta_p, delta_h))
solvents = np.asarray(solvents)

SM = [['IDIC',18.7,7.2,4.5,1011,1]]
polymers = [['PTB7-Th',18.56,2.3,3.21,5e4,1]]

SM_df = pd.DataFrame.from_records(SM, columns=['name','dD','dP','dH','MW', 'rho'])
polymer_df = pd.DataFrame.from_records(polymers, columns=['name','dD','dP','dH','MW', 'rho'])
solvents_df = pd.DataFrame.from_records(np.hstack((np.arange(len(solvents)).reshape(-1,1), solvents)),
                                        columns=['name','dD','dP','dH'])

def get_system(indx):
    delta_solvent = solvents_df.loc[indx[0],['dD','dP','dH']].tolist()
    M_solv = 1
    MVol = 100

    delta_sm = SM_df.loc[indx[1],['dD','dP','dH']].tolist()
    M_sm = (SM_df.loc[indx[1],'MW']/SM_df.loc[indx[1],'rho'])*(1/MVol)
    
    delta_polymer = polymer_df.loc[indx[2],['dD','dP','dH']].tolist()
    M_polymer = (polymer_df.loc[indx[2],'MW']/polymer_df.loc[indx[2],'rho'])*(1/MVol)
    
    M = [M_polymer, M_sm, M_solv]
    chi,_ = polyphase.get_chi_vector([delta_polymer,delta_sm,delta_solvent], MVol, 2)
 
    out = [polymer_df.loc[indx[2],'name'],SM_df.loc[indx[1],'name'],solvents_df.loc[indx[0],'name'],
           chi[0], chi[1], chi[2], M,
           delta_polymer, delta_sm,  delta_solvent
    ]   
    return out

axes = [np.arange(0,len(solvents_df)),np.arange(0,len(SM_df)),np.arange(0,len(polymer_df))]
df = pd.DataFrame(get_system(i) for i in product(*axes))
df.columns =['polymer', 'SM','solvent','chi12','chi13','chi23',
                 'dop','delta_polymer','delta_SM','delta_solvent'] 
print('Total of {} systems'.format(len(df)))

dirname = '../figures/deltaspace/{}/'.format(RUN_IDENTIFIER)
if os.path.exists(dirname):
    shutil.rmtree(dirname)    
os.makedirs(dirname)

num_simplices = []
for i, row in df.iterrows():
    M = row['dop']
    chi = [row['chi12'], row['chi13'], row['chi23']]
    f = lambda x: polyphase.flory_huggins(x , M, chi)
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
x = np.vstack(df['delta_solvent'].to_numpy())
sm = df['delta_SM'][0]
polymer = df['delta_polymer'][0]
for i,ax in enumerate(axs):
    path = ax.scatter(x[:,0], x[:,1], x[:,2],c=cvalues[:,i],
                      alpha=1.0, cmap='bwr')
    fig.colorbar(path, pad=0.05, ax = ax, fraction=0.035)
    
    ax.scatter(sm[0], sm[1], sm[2], 
               marker='s',s=100,color='k',label='small molecule' if i==2 else '')
    ax.scatter(polymer[0], polymer[1], polymer[2],
               marker='*',s=100,color='k',label='polymer' if i==2 else '')
    ax.set_xlabel(r'$\delta_{D}$')
    ax.set_ylabel(r'$\delta_{P}$')
    ax.set_zlabel(r'$\delta_{H}$')
    ax.set_title('{}-phase'.format(i+1))
    touchup3d(ax)
fig.legend()

plt.savefig('../figures/deltaspace/{}.png'.format(RUN_IDENTIFIER), dpi=500, bbox_inches='tight')
  
    










