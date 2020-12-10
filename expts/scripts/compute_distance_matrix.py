import pickle
import re
import pdb
import os
import numpy as np
import glob

from skimage.transform import resize
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray

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
    img0 = imread(file0)
    img1 = imread(file1)
    img0 = resize(img0, (64,64))
    img1 = resize(img1, (64,64))
    img0= rgb2gray(img0)
    img1= rgb2gray(img1)
    d = ssim(img0, img1)
    
    return 1-d


files = glob.glob("../figures/chispace/dimred/*.png")
M = get_distance_matrix(files, get_ssim_distance) 
print('Distance matrix has shape: {}'.format(M.shape))
out = {'M':M}
save_file = './data/chispace_dimred_mp10.pkl'
with open(save_file, 'wb') as fp:
    pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    