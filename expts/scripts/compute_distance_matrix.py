import pickle
import re
import pdb
import os
import numpy as np
import glob

import numpy as np
import pandas as pd

from skimage.transform import resize
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as MSE
from skimage.color import rgb2gray,rgba2rgb

from polyphase.parallel import get_distance_matrix

import ray
ray.init(address='auto', _node_ip_address=os.environ["ip_head"].split(":")[0],
         _redis_password=os.environ["redis_password"])
num_nodes = len(ray.nodes())
print('Total number of nodes are {}'.format(num_nodes))

def get_ssim_distance(file0, file1):
    """ Given two image file names in file0, file1 compute distance"""
    
    img0 = imread(file0)
    img1 = imread(file1)
    img0 = resize(img0, (64,64))
    img1 = resize(img1, (64,64))
    d = ssim(img0, img1, multichannel=True)
    
    return 1-d

def get_grayscale_ssim(file0, file1):
    """Given two file names, return SSIM distance
    
    file0, file1 : Image file address
    
    returns : SSIM similarity score 
    """
    img0 = imread(file0)
    img1 = imread(file1)
    img0 = resize(img0, (64,64))
    img1 = resize(img1, (64,64))
    img0= rgb2gray(rgba2rgb(img0))
    img1= rgb2gray(rgba2rgb(img1))
    d = ssim(img0, img1)
    
    return 1-d

def get_MSE(file0, file1):
    img0 = imread(file0)
    img1 = imread(file1)
    img0 = resize(img0, (64,64))
    img1 = resize(img1, (64,64))
    d = MSE(img0, img1)
    return d


def get_batch_of_phasediags(df, smstr, polymerstr):
    sys_df = df[(df['SM']==smstr) & (df['polymer']==polymerstr)]
    filename_list = []
    for _,row in sys_df.iterrows():
        fname = '../figures/hteplotsV2/{}_{}_{}.png'.format(row['solvent'], row['SM'], row['polymer'])
        filename_list.append(fname)
    
    return filename_list, sys_df

# Compute PM6-Y6-86 solvents distance matrix
htedf = pd.read_pickle('../data/htev2.pkl')
smstr = 'Y6'
polymerstr  = 'PM6'
files, sys_df = get_batch_of_phasediags(htedf, smstr, polymerstr)
print('Total of {} phase diagrams'.format(len(files)))

M = get_distance_matrix(files, get_MSE) 
print('Distance matrix has shape: {}'.format(M.shape))
out = {'M':M, 'df':sys_df}
save_file = './data/final/MSE_distance_PM6_Y6.pkl'
with open(save_file, 'wb') as fp:
    pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

# Compute distance matrix of the synthetic example    
files = glob.glob("../figures/chispace/dimred/*.png")
M = get_distance_matrix(files, get_MSE) 
print('Distance matrix has shape: {}'.format(M.shape))
out = {'M':M}
save_file = './data/final/MSE_chispace_dimred_mp10.pkl'
with open(save_file, 'wb') as fp:
    pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)   
    
    
    