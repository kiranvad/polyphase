import glob
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform, euclidean
import ray
import time
from collections import Counter
import torch
from PIL import Image
import torchvision.transforms as transforms
import re
from sklearn.manifold import TSNE
import seaborn as sns
import os
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt

""" The following is required for multinode parallelization """
print(os.environ["ip_head"], os.environ["redis_password"])
ray.init(address='auto', node_ip_address=os.environ["ip_head"].split(":")[0],redis_password=os.environ["redis_password"])

print("Number of Nodes in the Ray cluster: {}".format(len(ray.nodes())))

@ray.remote
def get_distance_row(data, metric, rowid, triuids):
    rowid_flags = triuids[0]
    rows = triuids[0][rowid_flags==rowid]
    cols = triuids[1][rowid_flags==rowid]
    dist_row = []
    for r,c in zip(rows, cols):
        dist_row.append(metric(data[r,:], data[c,:]))
    print('Computed {} on {}'.format(rowid, ray.services.get_node_ip_address()))
    
    return dist_row, rowid

def get_distance_matrix(X, metric):
    ray.init(ignore_reinit_error=True)
    start = time.time()
    n_samples, n_features = X.shape
    nC2 = 0.5*(n_samples*(n_samples-1))
    iu = np.triu_indices(n_samples,1)
    row_ids = np.unique(iu[0])
    
    iu_ray = ray.put(iu)
    X_ray = ray.put(X)
    metric_ray = ray.put(metric)
    
    remaining_result_ids = [get_distance_row.remote(X_ray, metric_ray, rowid, iu_ray) for rowid in reversed(row_ids)]
    
    dist = {}
    while len(remaining_result_ids) > 0:
        ready_result_ids, remaining_result_ids = ray.wait(remaining_result_ids, num_returns=1)
        result_id = ready_result_ids[0]
        dist_result,rowid_result = ray.get(result_id)
        dist[rowid_result] = dist_result
        print('Processed : {}'.format(rowid_result)) 
        
    del remaining_result_ids,result_id, iu_ray, metric_ray, X_ray      
    
    reduce_dist = [dist[k] for k in sorted(dist)]
    dist = np.hstack(reduce_dist)

    assert nC2==len(dist), "Not all the reduced distances are returned. expected {} got {}".format(nC2, len(dist))
    
    D = squareform(dist)
    
    assert n_samples==np.shape(D)[0] , "Shape of distance matrix is {}, expected {}x{}".format(D.shape, n_samples, n_samples)
    
    end = time.time()
    print('Computation took : {:.2f} sec'.format(end-start))

    return D

def distance_image(candidate, query):
    img1 = img2tensor(candidate[0])
    img2 = img2tensor(query[0])
    diff = img1 - img2
    dist = norm(diff)
    
    return dist

def img2tensor(image_file):
    pil_image = Image.open(image_file)
    p = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
    rgb_image_tensor = p(pil_image)[:3,:,:]
    
    return rgb_image_tensor

def imshow_tensor(rgb_image_tensor):
    plt.imshow(rgb_image_tensor.permute(1, 2, 0))
    plt.axis('off')
    plt.show()



images_dir = '../figures/hteplots/*.png'

images_list = sorted([file for file in glob.glob(images_dir)])

print('Total of {} phase diagrams'.format(len(images_list)))


X = np.asarray(images_list).reshape(-1,1)
M = get_distance_matrix(X, distance_image) 
print('Distance matrix has shape: {}'.format(M.shape))
np.save('../notebooks/hte_distmat.npy', M)

tagger = re.compile('../figures/hteplots/(.*)_(.*)_(.*).png')

tags_list = []
for img in images_list:
    tag = tagger.findall(img)
    tags_list.append(tag[0])
    
tags_array = np.asarray(tags_list)
print('tags array has the shape : {}'.format(tags_array.shape))

X_emb = TSNE(n_components=2, metric='precomputed').fit_transform(M)
np.save('../notebooks/hte_tsne_Xembed.npy', X_emb)


sns.scatterplot(x=X_emb[:,0],y= X_emb[:,1], hue = tags_array[:,0], palette="Set2")
plt.savefig('../figures/notebooks/tsne_hte_solvent.png', dpi=400)
plt.close()
sns.scatterplot(x=X_emb[:,0],y= X_emb[:,1], hue = tags_array[:,1], palette="Set2")
plt.savefig('../figures/notebooks/tsne_hte_smallmolecule.png', dpi=400)
plt.close()
sns.scatterplot(x=X_emb[:,0],y= X_emb[:,1], hue = tags_array[:,2], palette="Set2")
plt.savefig('../figures/notebooks/tsne_hte_polymer.png', dpi=400)
plt.close()

ray.shutdown()
