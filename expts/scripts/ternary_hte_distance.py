import numpy as np
import ray
import time
import re
import os

# skimage functions for distance measure
from skimage.transform import resize
from skimage.io import imread, imread_collection, imshow
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


from polyphase import timer
from polyphase.parallel import get_distance_matrix

""" The following is required for multinode parallelization """
print(os.environ["ip_head"], os.environ["redis_password"])
ray.init(address='auto', node_ip_address=os.environ["ip_head"].split(":")[0],redis_password=os.environ["redis_password"])

print("Number of Nodes in the Ray cluster: {}".format(len(ray.nodes())))


def get_ssim_distance(img0, img1):
    """ Given two images in img0, img1 compute distance"""
    
    img0 = resize(img0, (64,64))
    img1 = resize(img1, (64,64))
    d = ssim(img0, img1, multichannel=True)
    
    return d

images_dir = '../figures/hteplots/*.png'
images = imread_collection(images_dir)

print('Total of {} phase diagrams'.format(len(images)))

M = get_distance_matrix(images, get_ssim_distance) 
print('Distance matrix has shape: {}'.format(M.shape))
np.save('../data/hte_distmat_ssim.npy', M)


ray.shutdown()
