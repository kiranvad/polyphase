import pdb
import numpy as np
import ray
from ..helpers import timer

from scipy.spatial.distance import squareform

@ray.remote
def _get_distance_row(data, metric, rowid, triuids):
    print('Staging {} on {}'.format(rowid, ray.services.get_node_ip_address()))
    T = timer()
    rowid_flags = triuids[0]
    rows = triuids[0][rowid_flags==rowid]
    cols = triuids[1][rowid_flags==rowid]
    dist_row = []
    for r,c in zip(rows, cols):
        dist_row.append(metric(data[r], data[c]))

    return dist_row, rowid, T.end()

def get_distance_matrix(X, metric):
    """ Compute distance matrix in parallel using ray
    
    Computes pairwise distances between samples with arbitrary dimensions.
    a `metric` needs to be passed as a callable function that takes two samples 
    of the data `X` and returns a scalar distance.
    
    Example:
    --------
    from scipy.spatial.distance import euclidean
    import numpy as np
    
    X = np.random.rand(4,3)
    M = get_distance_matrix(X, euclidean) 
    # M should be of the shape (4,4)
    
    Input:
    ------
        X       :  data matrix where each row is a sample
        metric  :  A metric function that is used to compute distance
        
    Output:
    -------
        M    : Distance matrix in squareform
        
        
    Notes:
    ------
    When you pass large numpy arrays, this function might run into memory issues. One work around is to pass 
    a small list that can be used to query samples by calling `__getitem__`
    
    """
    ray.init(ignore_reinit_error=True)
    T = timer()
    n_samples = len(X)
        
    nC2 = 0.5*(n_samples*(n_samples-1))
    iu = np.triu_indices(n_samples,1)
    row_ids = np.unique(iu[0])
    
    iu_ray = ray.put(iu)
    X_ray = ray.put(X)
    metric_ray = ray.put(metric)
    
    remaining_result_ids = [_get_distance_row.remote(X_ray, metric_ray, rowid, iu_ray) for rowid in reversed(row_ids)]
    
    dist = {}
    while len(remaining_result_ids) > 0:
        ready_result_ids, remaining_result_ids = ray.wait(remaining_result_ids, num_returns=1)
        result_id = ready_result_ids[0]
        dist_result,rowid_result, time_result = ray.get(result_id)
        dist[rowid_result] = dist_result
        print('Processed : {} took {} '.format(rowid_result, time_result)) 
        
    del remaining_result_ids,result_id, iu_ray, metric_ray, X_ray, dist_result,rowid_result, time_result      
    
    reduce_dist = [dist[k] for k in sorted(dist)]
    dist = np.hstack(reduce_dist)

    assert nC2==len(dist), "Not all the reduced distances are returned. expected {} got {}".format(nC2, len(dist))
    
    D = squareform(dist)
    
    assert n_samples==np.shape(D)[0] , "Shape of distance matrix is {}, expected {}x{}".format(D.shape, n_samples, n_samples)
    
    print('\nComputation took : {}'.format(T.end()))

    return D
