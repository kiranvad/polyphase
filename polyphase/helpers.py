from math import pi
import numpy as np
import pdb

def get_ternary_coords(point):
    a,b,c = point
    x = 0.5-a*np.cos(pi/3)+b/2;
    y = 0.866-a*np.sin(pi/3)-b*(1/np.tan(pi/6)/2);
    
    return [x,y]

# Flory Huggins free energy with Beta correction term
import autograd.numpy as anp
from autograd import jacobian

def flory_huggins(x, M,CHI,beta=1e-3):
    T1 = 0
    for i,xi in enumerate(x):
        T1 += (xi*anp.log(xi))/M[i] + beta/xi
    T2 = 0.5*anp.matmul((anp.matmul(x,CHI)),anp.transpose(x)) 
    
    return T1+T2  

def FHTaylor(x,M,CHI, beta=1e-3,n=25):
    T1 = 0
    for i,xi in enumerate(x):
        T1 += (xi*log(xi,n=n))/M[i] + beta/xi
    T2 = 0.5*anp.matmul((anp.matmul(x,CHI)),anp.transpose(x)) 
    
    return T1+T2

def log(x,n=25):
    """
    Computes log(x) = \Sum_{i=1}^{\infty} [(-1^{i+1}/i)*(x-1)^i]
    """
    val = 0
    for i in range(1,n):
        val += (pow(-1,i+1)*pow(x-1,i))/i
        
    return val
 
def from4d23d(fourd_coords):
    x,y,z,w = fourd_coords
    u = y+0.5*(z+w)
    v = np.sqrt(3)*(z/2 + w/6) 
    w = np.sqrt(6)*(w/3)
    
    return [u,v,w]

# determine points inside a polyhedron
from scipy.spatial import Delaunay

def inpolyhedron(ph,points):
    """
    Given a polyhedron vertices in `ph`, and `points` return 
    critera that each point is either with in or outside the polyhedron
    
    Both polyhedron and points should have the same shape i.e. num_points X num_dimensions
    
    Returns a boolian array : True if inside, False if outside
    
    """
    tri = Delaunay(ph)
    inside = Delaunay.find_simplex(tri,points)
    criteria = inside<0
    return ~criteria

def get_convex_faces(v):
    verts = [ [v[0],v[1],v[3]], [v[1],v[2],v[3]],\
             [v[0],v[2],v[3]], [v[0],v[1],v[2]]]
    return verts

def utri2mat(utri, dimension):
    """ convert list of chi values to a matrix form """
    inds = np.triu_indices(dimension,1)
    ret = np.zeros((dimension, dimension))
    ret[inds] = utri
    ret.T[inds] = utri

    return ret