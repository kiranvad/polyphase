import os
import numpy as np
import pandas as pd
import pickle
from math import floor, ceil
import glob
import time

try:
    from sklearn.manifold import MDS, Isomap
except ImportError as exc:
    raise ImportError('This analysis requires scikit-learn module\n'
                     'Install it using pip install sklearn')

from sklearn.cluster import SpectralClustering

from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from scipy.sparse import csgraph
from scipy.linalg import eigvalsh

import polyphase as phase

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True})
import matplotlib.colors as mcolors
tab_colors = list(mcolors.TABLEAU_COLORS)


def touchup3d(ax):
    """Modify the aesthetics of the 3D plot
    
    A utility function to modify the 3D plot to look neater
    ax : a matplotlib 3D axis
    """
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    ax.zaxis._axinfo['juggled'] = (1,2,0)

"""
A set of helper functions to set up a pipeline for clustering analysis.
"""

def _get_explained_variance(Dm,emb):
    """Compute explained variance for an embedding using metric
    Inputs:
    =======
        Dm. : Metric as a matrix of shape (n_samples, n_samples)
        emb : Embedding as an array of shape (n_samples, dimension)
        
    Output:
    =======
        explained_variance. : Explained variance as an list of length 3.
                              each entry in the list corresponds to variances contained in the dimensions upto it
    
    Here and throughout this file, n_samples refers to number of samples and dimension
    to refers to dimension of the data considered
    
    see : http://web.mit.edu/cocosci/Papers/sci_reprint.pdf
    """
    explained_variance = []
    for dim in [1,2,3]:
        Dy = squareform(pdist(emb[:,:dim]))
        explained_variance.append(pearsonr(Dm.flatten(), 
                                           Dy.flatten())[0])
    
    return explained_variance
    
def perform_isomap(M):
    """Utility function to perform Isomap based on a metric
    
    Input:
    ======
         M    :  Metric as a matrix of shape (n_samples, n_samples)
    Output:
    =======
        emb_isomap  :  Embedding of the resulting Isomap algorithm as an array of shape (n_samples, 3)
        explained_variance : Output from the function `_get_explained_variance` above
    """
    embedding = Isomap(n_components=3,n_neighbors=5, 
                       metric='precomputed')
    embedding.fit(M)
    Dm = embedding.dist_matrix_
    emb_isomap = embedding.embedding_
    explained_variance = _get_explained_variance(Dm,emb_isomap)
        
    return emb_isomap, explained_variance

def perform_MDS(M):
    """Utility function to perform Multi-Dimensional Scaling (metric MDS) based on a metric
    
    Input:
    ======
         M    :  Metric as a matrix of shape (n_samples, n_samples)
    Output:
    =======
        emb  :  Embedding of the resulting MDS algorithm as an array of shape (n_samples, 3)
        explained_variance : Output from the function `_get_explained_variance` above
    """
    embedding = MDS(n_components=3,  dissimilarity='precomputed', 
                    metric=True, random_state=0)
    emb = embedding.fit_transform(M)
    explained_variance = _get_explained_variance(M,emb)
    
    return emb,explained_variance

# 5. Perform clustering
def cluster_embedding(X, emb, n_clusters=4):
    """Given a low dimensional embedding, cluster them and return labels,
    eigen values of graph Laplacian
    
    X : affinity matrix of images
    emb : embedding (array of shape (num_points, dimension))
    n_clusters : Number of clusters expected
    
    returns:
        higdim_labels : Labels for the clustering performed on images addinity matrix
        lowdim_labels : Labels for the points in `emb` based on clustering
        eigen_values. : Eigen values of the graph Laplacian
        
    """
    clustering = SpectralClustering(n_clusters=n_clusters,assign_labels="discretize",
                                    random_state=0, affinity='precomputed')
    
    highdim_labels = clustering.fit_predict(X)
    
    D = squareform(pdist(emb, 'euclidean'))
    delta = 0.01
    A = np.exp(- D ** 2 / (2. * delta ** 2))
    lowdim_labels = clustering.fit_predict(A)
    
    return highdim_labels, lowdim_labels

class Pipeline:
    """A pipeline class used for cluster analysis
    Inputs:
    =======
        M.  :  Metric as a matrix of shape (n_samples, n_samples)
        
    Methods:
    ========
        compute         :  Main method to perform dimensionality reduction and clustering
        plot_embedding  :  Plots embedding as in the Euclidean space
        show_clusters   :  Visualize the phase diagrams arranged by clustered indices 
        
    
    Attributes:
    ===========
        X.                  : Gaussian similarity measure computed
        emb.                :  Embedding obtained when calling `.compute` method
        explained_variance  : Explained variance of the method used in `.compute`
        highdim_labels      : Labels of spectral clustering method with metric on the original data space
        lowdim_labels       : Labels of spectral clustering method with metric on the embedding Euclidean data space
    
    """
    def __init__(self,M):
        self.M = M
        delta = M.std()
        self.X = np.exp(- M ** 2 / (2. * delta ** 2))
        
    def compute(self, drmethod='isomap', n_clusters=4):
        if drmethod=='isomap':
            self.emb, self.explained_variance = perform_isomap(self.M)
        elif drmethod=='mds':
            self.emb, self.explained_variance = perform_MDS(self.M)
            
        self.highdim_labels, self.lowdim_labels = cluster_embedding(self.X, self.emb,
                                                                        n_clusters=n_clusters)
        
        return
    
    def plot_embedding(self, use_highdim=True, ax = None):
        """
        Inputs:
        =======
            use_highdim  : (boolean) Whether to use clustering labels from original data space 
                           or the embedding (default, True)
            ax           : Matplotlib ax (default, None i.e. generated within the class)
            
        Outputs:
        ========
            handles of the current figure (fig) and axis (ax)
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
            
        if use_highdim:
            _labels = self.highdim_labels
        else:
            _labels = self.lowdim_labels    
            
        if hasattr(self, 'emb'):
            all_labels = np.unique(_labels)
            for i, label in enumerate(all_labels) :
                ax.scatter(self.emb[_labels==label,0], 
                           self.emb[_labels==label,1], 
                           label=str(label), color=tab_colors[i])
            ax.legend()                                                                                            
            ax.set_xlabel(r"Coordinate 1 ({:.2f}\%)".format(self.explained_variance[0]*100))
            ax.set_ylabel(r"Coordinate 2 ({:.2f}\%)".format((self.explained_variance[1]
                                                             -self.explained_variance[0])*100))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            return ax, fig
        else:
            RuntimeError('You need to call .compute(*,**) before .plot_embedding(*,**)')
            
    def show_clusters(self,files, use_highdim=True):
        """
        Inputs:
        =======
            use_highdim  : (boolean) Whether to use clustering labels from original data space 
                           or the embedding (default, True)
            files        : location of the folder with phase diagrams
        """
        
        if use_highdim:
            _labels = self.highdim_labels
        else:
            _labels = self.lowdim_labels
            
        ic = glob.glob(files)
        num_images = len(ic)

        # Two pairs of `nrows, ncols` are possible
        k = (num_images * 12)**0.5
        r1 = max(1, floor(k / 4))
        r2 = ceil(k / 4)
        c1 = ceil(num_images / r1)
        c2 = ceil(num_images / r2)

        # Select the one which is closer to 4:3
        if abs(r1 / c1 - 0.75) < abs(r2 / c2 - 0.75):
            nrows, ncols = r1, c1
        else:
            nrows, ncols = r2, c2

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows,ncols))
        ax = np.asarray(axes).ravel()
        all_labels = np.unique(_labels)
        axis_id = 0
        for label in all_labels:
            label_files = np.where(_labels==label)[0]
            print('Cardinality of cluster {} is {}'.format(label, len(label_files)))
            for n in label_files:
                img = plt.imread(files.replace('*',str(n)))
                ax[axis_id].imshow(img)
                ax[axis_id].axis('off')
                ax[axis_id].set_title('{}'.format(label))
                axis_id += 1
        for i in range(axis_id,int(nrows*ncols)):
            ax[i].axis('off')
        
        return fig    
    
    
# 1. Apply the workflow to synthetic data set of chispace exploration
dirname = './data/pm6y6'
distmats = dirname + '/distance.pkl'
with open(distmats, 'rb') as handle:
    out = pickle.load(handle)
    
USE_HIGHDIM = True
NUM_CLUSTERS = 5 # a hyper-parameter querying number of clusters from clustering algorithm
sys_df = out['df'].reset_index(drop=True)
sm = sys_df['delta_SM'][0]
polymer = sys_df['delta_polymer'][0]

M = out['M']
workflow = Pipeline(M)
workflow.compute(drmethod='mds', n_clusters=NUM_CLUSTERS)
print('**Computing the embedding**')
workflow.plot_embedding(use_highdim=USE_HIGHDIM)
sys_df['cluster'] = workflow.highdim_labels
plt.savefig('embedding.png')

print('**Generating the cluster labelled phase diagram plot**')
fig = workflow.show_clusters(dirname+'/pds/*.png',use_highdim=USE_HIGHDIM)
plt.savefig('clusters.png')

# 2. Visualize clusters in the design space
print('**Generating design space visualization of clusters**')
deltas = np.asarray(sys_df['delta_solv'].to_list())
MVols = np.asarray(sys_df['MVol'].to_list())

fig, ax = plt.subplots(figsize=(5,5), subplot_kw={'projection':'3d'})
labels = workflow.highdim_labels
unique_labels = np.unique(labels)
for ul in unique_labels:
    ax.scatter(deltas[labels==ul,0], deltas[labels==ul,1],
              deltas[labels==ul,2], label=str(ul), s = MVols[labels==ul])
    
ax.scatter(sm[0], sm[1], sm[2], 
           marker='s',s=100,color='k',label='small molecule')
ax.scatter(polymer[0], polymer[1], polymer[2],
           marker='*',s=100,color='k',label='polymer') 

ax.set_xlabel(r'$\delta_{D}$')
ax.set_ylabel(r'$\delta_{P}$')
ax.set_zlabel(r'$\delta_{H}$')

fig.legend(ncol=2)
touchup3d(ax)
plt.savefig('designspace.png')

print('**Total system + CPU time is {:.2f} seconds**'.format(time.process_time()))

"""
To apply the approach to the model dataset, change the dirname to './data/chispace'
Note that the design space needs to plotted accordingly.
"""
