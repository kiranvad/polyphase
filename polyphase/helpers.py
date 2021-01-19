from math import pi
import numpy as np
import pdb
from scipy.spatial import Delaunay

def get_ternary_coords(point):
    """ Compute 2d embedding of a 3d hyperplane """
    a,b,c = point
    x = 0.5-a*np.cos(pi/3)+b/2;
    y = 0.866-a*np.sin(pi/3)-b*(1/np.tan(pi/6)/2);
    
    return [x,y]

def from4d23d(fourd_coords):
    """ Compute 3d embedding of a 4d hyperplane """
    x,y,z,w = fourd_coords
    u = y+0.5*(z+w)
    v = np.sqrt(3)*(z/2 + w/6) 
    w = np.sqrt(6)*(w/3)
    
    return [u,v,w]


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

import time

class timer:
    def __init__(self):
        self.start = time.time()
    def end(self):
        end = time.time()
        hours, rem = divmod(end-self.start, 3600)
        minutes, seconds = divmod(rem, 60)

        return "{:0>2} Hr:{:0>2} min:{:05.2f} sec".format(int(hours),int(minutes),seconds)
      