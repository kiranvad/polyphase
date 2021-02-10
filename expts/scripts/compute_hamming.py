import pickle, re, pdb, os, glob, shutil
from itertools import product

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform, hamming

import polyphase
import matplotlib.pyplot as plt

class Store:
    def __init__(self):
        self.storage = []
 
    def save(self, array):
        self.storage.append(array)
        
    def __getitem__(self,index):
        return self.storage[index]
    
    def __len__(self):    
        return len(self.storage)

def get_batch_of_phasediags(df, smstr, polymerstr):
    sys_df = df[(df['SM']==smstr) & (df['polymer']==polymerstr)]
    filename_list = []
    for _,row in sys_df.iterrows():
        fname = '../figures/hteplotsV2/{}_{}_{}.png'.format(row['solvent'], row['SM'], row['polymer'])
        filename_list.append(fname)
    
    return filename_list, sys_df

def distance_hamming(data,i,j):
    x = data[i[0]]
    y = data[j[0]]
    
    return hamming(x,y)

# """
# 1. Compute PM6-Y6-86 solvents distance matrix
# """

# dirname = './data/hamming/pm6y6'
# if os.path.exists(dirname):
#     shutil.rmtree(dirname)
# os.makedirs(dirname)
# os.makedirs(dirname + '/pds')

# htedf = pd.read_pickle('../data/htev3.pkl')
# smstr = 'Y6'
# polymerstr  = 'PM6'
# _, sys_df = get_batch_of_phasediags(htedf, smstr, polymerstr)
# sys_df = sys_df.reset_index()

# data_pm6y6 = Store()
# for i,row in sys_df.iterrows():
#     chi = [row['chi12'], row['chi13'], row['chi23']]
#     M = row['dop']
#     f = lambda x : polyphase.flory_huggins(x, M, chi)
#     engine = polyphase.PHASE(f,100,3)
#     engine.compute(use_parallel=False, verbose=False, lift_label=True)
    
#     y = engine.df.loc['label',:].to_numpy().astype('int')
#     data_pm6y6.save(y)
    
#     polyphase.plain_phase_diagram(engine.df)
#     plt.savefig(dirname+'/pds/{}.png'.format(i), bbox_inches='tight', dpi=300)
#     plt.close()
    
# dist = lambda x,y: distance_hamming(data_pm6y6,x,y)
# M = squareform(pdist(np.arange(len(data_pm6y6)).reshape(-1,1), dist))
# print('Distance matrix has shape: {}'.format(M.shape))
# out = {'M':M, 'df':sys_df}
# save_file = dirname + '/distance.pkl'
# with open(save_file, 'wb') as fp:
#     pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

    
"""
2. Compute distance matrix of the synthetic example   
"""
dirname = './data/hamming/chispace'
if os.path.exists(dirname):
    shutil.rmtree(dirname)
os.makedirs(dirname)
os.makedirs(dirname+ '/pds')

Mp = 10
chi12 = np.linspace(1.3,1.44, num=6)
chi13 = [0.3,0.6,1,1.5, 2,2.5, 3]
chi23 = [0.3,0.6,1,1.5, 2,2.5, 3]
chispace = list(product(chi12, chi13, chi23))
chispace = np.asarray(chispace)
df = pd.DataFrame(chispace, columns=['chi12', 'chi13', 'chi23'])

num_simplices = []
data_synth = Store()
for i, chi in df.iterrows():
    M = [Mp,10,1]
    f = lambda x: polyphase.flory_huggins(x , M, chi)
    engine = polyphase.PHASE(f, 100,3)
    engine.compute(use_parallel=False, verbose=False, lift_label=True)
    y = engine.df.loc['label',:].to_numpy().astype('int')
    data_synth.save(y)
    
    polyphase.plain_phase_diagram(engine.df)
    #fname = '_'.join('{}'.format(i) for i in chi).replace('.','p')
    plt.savefig(dirname+'/pds/{}.png'.format(i), bbox_inches='tight', dpi=300)
    plt.close()

dist = lambda x,y: distance_hamming(data_synth,x,y)
M = squareform(pdist(np.arange(len(data_synth)).reshape(-1,1), dist))
print('Distance matrix has shape: {}'.format(M.shape))
out = {'M':M, 'df':df}
save_file = dirname + '/distance.pkl'
with open(save_file, 'wb') as fp:
    pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)   
    
    
    