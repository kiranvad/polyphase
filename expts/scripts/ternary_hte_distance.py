import numpy as np
import ray
import time
import re
import os
import pickle
import pandas as pd
import itertools

# skimage functions for distance measure
from skimage.transform import resize
from skimage.io import imread, imread_collection, imshow
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# from polyphase
from polyphase import timer
from polyphase.parallel import get_distance_matrix

""" The following is required for multinode parallelization """

print(os.environ["ip_head"], os.environ["redis_password"])
ray.init(address='auto', node_ip_address=os.environ["ip_head"].split(":")[0],redis_password=os.environ["redis_password"])

print("Number of Nodes in the Ray cluster: {}".format(len(ray.nodes())))


""" Start computation from here """
allsys_df = pd.read_pickle('../expts/data/allsys_df.pkl')
allsys_df.head()


def get_batch_of_phasediags(smstr, polymerstr):
    """
    
    Given a pair of small molecule and polymer, 
    returns portion of data frame with phase diagrams containing the given pair
    
    """
    sys_df = allsys_df[(allsys_df['SM']==smstr) & (allsys_df['polymer']==polymerstr)]
    filename_list = []
    for _,row in sys_df.iterrows():
        fname = '../figures/hteplots/{}_{}_{}.png'.format(row['solvent'], row['SM'], row['polymer'])
        filename_list.append(fname)
    
    return filename_list, sys_df

def get_ssim_distance(file0, file1):
    """ 
    Given two image file names in file0, file1 compute distance
    """
    
    img0 = imread(file0)
    img1 = imread(file1)
    img0 = resize(img0, (64,64))
    img1 = resize(img1, (64,64))
    d = ssim(img0, img1, multichannel=True)
    
    return 1-d

def get_multiple_systems(systems):
    """ 
    Given a list of (small molecule, polymer) tuples, 
    returns portion of the dataframe with the phase diagrams of the system
    
    """
    files, sys_df = [], []
    for indx, (i,j) in enumerate(systems):
        _files, _sys_df = get_batch_of_phasediags(i, j)
        files.append(_files)
        sys_df.append(_sys_df)
        print(indx, i, j)

    sys_df = pd.concat(sys_df)
    files = list(itertools.chain.from_iterable(files))
    
    assert len(sys_df)==86*len(systems), "Expected {} phase diagrams got {}".format(86*len(systems),len(sys_df))
    
    return files, sys_df

systems = [('PC61BM','PM6'), ('Y6', 'PM6')]
x = [item for t in systems for item in t] 
fname = '_'.join(xi for xi in x)

files, sys_df = get_multiple_systems(systems)

print('Total of {} phase diagrams'.format(len(files)))

M = get_distance_matrix(files, get_ssim_distance) 
print('Distance matrix has shape: {}'.format(M.shape))

out = {'df':sys_df,'M':M}
save_file = './data/hte_distance_{}.pkl'.format(fname)
with open(save_file, 'wb') as fp:
    pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

ray.shutdown()
