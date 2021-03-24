import numpy as np
import pdb
from itertools import combinations
from scipy.spatial import Delaunay
from scipy.constants import gas_constant

import time

class timer:
    def __init__(self):
        self.start = time.time()
    def end(self):
        end = time.time()
        hours, rem = divmod(end-self.start, 3600)
        minutes, seconds = divmod(rem, 60)

        return "{:0>2} Hr:{:0>2} min:{:05.2f} sec".format(int(hours),int(minutes),seconds)

def _utri2mat(utri, dimension):
    """ convert list of chi values to a matrix form """
    inds = np.triu_indices(dimension,1)
    ret = np.zeros((dimension, dimension))
    ret[inds] = utri
    ret.T[inds] = utri

    return ret

def _ln(x, thresh=1e-2):
    if x<thresh:
        return x-1
    else:
        return np.log(x)

def flory_huggins(x, M,chi,beta=0.0, logapprox=False):
    """ Free energy formulation 
    
    parameters:
    -----------
        x    :  Composition as a list (dim, )
        M    :  Degree of polymerization
        chi  :  flory-huggins interaction parameters as a list of (nCdim)
    
    Optional:
    ---------
        beta        : Coefficients the beta correction term (default, 0.0)
        logapprox   : Whether to use the approximation as log(x)=x-1 when x~0  
    
    """
    CHI = _utri2mat(chi, len(M))
    T1 = 0
    for i,xi in enumerate(x):
        if not logapprox:
            T1 += (xi*np.log(xi))/M[i] + beta/xi
        else:
            T1 += (xi*_ln(xi))/M[i] + beta/xi
    T2 = 0.5*np.matmul((np.matmul(x,CHI)),np.transpose(x)) 
    
    return T1+T2  
        
def polynomial_energy(x):
    """ Free energy using a polynomial function for ternary """
    
    assert len(x)==3,'Expected a ternary system got {}'.format(len(x))
    
    #e = (x[0]**2)*(x[1]**2) + (x[0]**2 + x[1]**2)*(x[2]**2)
    # e = -e/0.5
    e =0
    for xi in x:
        e += ((xi-0.1)**2)*((0.9-xi)**2)

    return e*1e3


""" Compute chi from solubilities """

def _compute_chi(delta_i,delta_j,V):
    """
    total solubility parameters delta_i, delta_j are computed from hydrogen, polar, dispersive components
    
    delta_i, delta_j in MPa^{1/2} and V in cm3/mol
    
    returns a scalar chi value
    
    """
    constant = 1.0 #4.184*(2.045**2)/(8.314)
    chi_ij =  0.34+(constant)*(V/(gas_constant*300)*( np.asarray(delta_i) - np.asarray(delta_j) )**2)
        
    return chi_ij

def _compute_weighted_chi(vec1,vec2,V, W):
    value = 0.0
    for i,w  in enumerate(W):
        value += w*(vec1[i]-vec2[i])**2
    
    value = 0.34 + value*(V/(gas_constant*300))
    
    return value
                   
def get_chi_vector(deltas, V0, approach=1):
    """
    Given a list of deltas, computes binary interactions of chis
    
    Chi values are ordered in the basis of the input deltas.
    For example, for a ternary system with deltas input as [polymer, sm, solvent]
    chi values are [(p,sm), (p, solv), (sm, solv)]
    """
    combs = combinations(deltas,2)
    inds = list((i,j) for ((i,_),(j,_)) in combinations(enumerate(deltas), 2))

    if approach==1:
        chi = [_compute_chi(np.linalg.norm(i[0]),np.linalg.norm(i[1]),V0) for i in combs]
    elif approach==2:
        chi = [_compute_weighted_chi(i[0],i[1],V0, W = [1.0,1.0,1.0]) for i in combs] 
    elif approach==3:
        chi = [_compute_weighted_chi(i[0],i[1],V0, W = [1.0,0.25,0.25]) for i in combs]            
    else:
        raise KeyError
        
    return chi, inds
    
def get_sample_data(ind):
    if ind==0:
        M = [5,5,1]
        chi = [1,0.5,0.5]
    if ind==1:
        M = [64,1,1]
        chi = [1,0.3,0.2]
    if ind==2:
        M = [1,1,1,1] 
        chi = 3.10*np.ones(int(0.5*4*(4-1)))
    
    return M,chi
    
    
""" utilities for some example runs """

def get_data(name='ow',fhid=0):
    """
    For a given data name and produces a M and Chi parameters of a ternary system.
    If the name is  FHPaper, it requires an id in fhid (default 0 ) in the range of (0,4)
    Get a dummy sets of data for trials:

    from solvers.utils import get_data
    M,CHI=get_data(name='ow',fhid=0)
    
    OW APS Physics Letters: https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=1231&context=me_pubs
    B Zhou Page:65 
    https://dspace.mit.edu/bitstream/handle/1721.1/35312/76904537-MIT.pdf?sequence=2&isAllowed=y
    
    """
    floryhuggins =[{'name': 'CHCl3/F8/PCBM','chi': [0.341,0.885,0.941]},
                  {'name': 'chlorobenzene/PFB/PCBM','chi': [0.340,0.899,0.899]},
                  {'name': 'CHCl3/APFO-3/PCBM','chi': [0.505,0.885,0.450]},
                  {'name': 'chlorobenzene/APFO-3/PCBM','chi': [0.480,0.899,0.479]}, 
                  {'name': 'xylene/PFB/PCBM','chi': [0.442,1.630,1.013]}
                   ]
    N = {'CHCl3':1,'chlorobenzene':1,'xylene':1,'F8':720,'PFB':73,'F8BT':172,'APFO-3':66,'PCBM':5}
    fname = name
    if name is 'ow':
        M = np.array([5,1,5])
        chi = [0.5,1,0.5] # chi_12, chi_13, chi_23
    elif name is 'bz65':
        M = np.array([1,1,64])
        chi = [0.2,0.3,1] # chi_12, chi_13, chi_23
    elif name is 'FHPaper':
        chiset = floryhuggins[fhid]
        solvent,polymer,PCBM = chiset['name'].split("/")
        M = [N[solvent],N[polymer],N[PCBM]]
        #M = np.array([1.0,64,5])
        chi = np.asarray(chiset['chi'])
        fname = "_".join([solvent,polymer,PCBM])
    elif name is 'temp':
        M = np.array([1.0,64.0,1.0])
        chi = [1.0,0.2,0.3] # chi_12, chi_13, chi_23
    else:
        raise KeyError('No such data exists')
    CHI = np.array([[0,chi[0],chi[1]],[chi[0],0,chi[2]],[chi[1],chi[2],0]])    
    
    info = {'params':r'M:{},$\chi$:{}'.format(M,chi),'fname':fname}
    
    return M, CHI, info
