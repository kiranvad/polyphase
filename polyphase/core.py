import pdb
import numpy as np
import time
import pandas as pd
import ray
from ._phase import (_serialcompute,
                     _parcompute, 
                     makegridnd,
                     is_boundary_point, is_pure_component,
                    get_max_delaunay_edge_length)
from scipy.spatial import Delaunay
from .visuals import TernaryPlot, QuaternaryPlot
import matplotlib.pyplot as plt

MIN_POINT_PRECISION = 1e-8
                     
class PHASE:
    def __init__(self,energy_func, meshsize,dimension):
        """Computing phase diagram using Convex Hull Method
        
        Parameters:
        -----------
            energy_func      :  (callable) Energy function that takes a d-dimensional list of 
                                compositions and returns a scalar energy
            meshsize         :  (int) Number of points to be sampled per dimension
            dimension        :  (int) Dimension of the the system 
        """
        if not callable(energy_func):
            raise ValueError('Vairable energy needs to be a function such as `polyphase.utils.flory_huggins`')
        self.energy_func = energy_func
        self.meshsize = meshsize
        self.dimension = dimension
        self.is_solved = False
    
    def in_simplex(self, point, simplex):
        """Find if a point is in a simplex
        returns True if the point is in simplex False otherwise
        """
        v = np.asarray([self.grid[:-1,x] for x in simplex])
        tri = Delaunay(v)
        inside = Delaunay.find_simplex(tri,point[:-1])

        if inside<0:
            return False
        else:
            return True

    def is_boundary_point(self,point):
        
        return is_boundary_point(point)

    def is_pure_component(self,point):
        
        return is_pure_component(point)
        
    def compute(self, **kwargs):
        """ Compute the phase diagram
        
        Arguments:
        ----------
            use_parallel       : (bool) whether to use a parallel computation (default, False)
            verbose            : (bool) whether to print more information as the computation progresses
            correction         : (string) Two types of corrections to energy surface are provided
                                        1. 'edge' -- where all the energy values near the boundary of
                                            hyperplane will be lifted to a constant energy (default)
                                        2. 'vertex' -- similar to 'edge' but the process is performed
                                            only for points at the vertices of hyperplane
            lift_label         : (bool) whether to interpolate the label of a simplex into points 
                                        inside it (default, True)
            refine_simplices   : (bool) whether to remove simplices that are on the "upper convex hull".
                                        (default, True) Note that the Gibbs criteria uses the lower 
                                        convex hull, thus it is recommended to set the
                                        simplex refinement to True
            thresholding       : (string) Two types of thresholding methods are implemented
                                          1. 'uniform' -- where the reference distance for number 
                                             of connected components of a simplex is computed 
                                             using the original grid length (default)
                                          2. 'delaunay'-- the length is computed using a delaunay 
                                              edge length of the initial mesh
                                              
            thresh_scale       : (float) scaling to be used for the edge length of the reference 
                                         in 'thresholding'
                                         
        Attributes:
        -----------
            grid         :  Grid used to compute the energy surface (array of shape (dim, points))
            energy       :  Free energy computed using self.energy_func (array of shape (points,))
            hull         :  scipy.spatial.ConvexHull instance of computed for energy landscape
            thresh       :  length scale used to compute adjacency matrix
            upper_hull   :  boolean flagg of each simplex in hull.simplices whether its a upper hull
            simplices    :  simplices of the lower convex hull of the energy landscape
            num_comps    :  connected components of each simplex as a list
            df           :  pandas.DataFrame with volume fractions and labels rows
            coplanar.    :  a list of boolean values one for each simplex (True- coplanar, False- not, None- Not computed)
        """
        
        self.use_parallel = kwargs.get('use_parallel', False)
        self.verbose = kwargs.get('verbose', False)
        self.correction = kwargs.get('correction', 'edge')
        self.lift_label = kwargs.get('lift_label',True)
        self.refine_simplices = kwargs.get('refine_simplices',True)
        self.thresholding = kwargs.get('thresholding','uniform')
        self.thresh_scale = kwargs.get('thresh_scale', 0.1*self.meshsize)
        _kwargs = self.get_kwargs()
        
        if self.use_parallel:
            outdict = _parcompute(self.energy_func, self.dimension, self.meshsize,**_kwargs)
        else:
            outdict = _serialcompute(self.energy_func, self.dimension, self.meshsize,**_kwargs)
        
        self.grid = outdict['grid'] 
        self.energy = outdict['energy'] 
        self.hull = outdict['hull'] 
        self.thresh = outdict['thresh'] 
        self.upper_hull = outdict['upper_hull']
        self.simplices = outdict['simplices']
        self.num_comps = outdict['num_comps'] 
        self.df = outdict['output']
        self.coplanar = np.asarray(outdict['coplanar'], dtype=bool)
        
        self.is_solved = True
        
        return

    def get_phase_compositions(self, point):
        """Compute phase contributions given a composition
        
        input:
        ------
            point : composition as a numpy array (dim,)
            
        output:
        -------
            x. : Phase compositions as a numpy array of shape (dim, )
            vertices  : Compositions of coexisting phases. Each row correspond to 
                        an entry in x with the same index
        """
        from scipy.optimize import lsq_linear
        
        if not self.is_solved:
            raise RuntimeError('Phase diagram is not computed\n'
                               'Use .solve() before requesting phase compositions')
            
        assert len(point)==self.dimension,'Expected {}-component composition got {}'.format(self.dimension, len(point))
        
        if self.is_boundary_point(point):
            raise RuntimeError('Boundary points are not considered in the computation.')
        
        inside = np.asarray([self.in_simplex(point, s) for s in self.simplices], dtype=bool)
        
        for i in np.where(inside)[0]:
            simplex = self.simplices[i]
            num_comps = np.asarray(self.num_comps)[i]
            vertices = self.grid[:,simplex]
            
            if num_comps==1:
                # no-phase splits if the simplex is labelled 1-phase
                continue

            A = np.vstack((vertices.T[:-1,:], np.ones(self.dimension)))
            b = np.hstack((point[:-1],1))
            
            lb = np.zeros(self.dimension)
            ub = np.ones(self.dimension)
            res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=0)
            x = res.x
            if not (x<0).any():
                # STOP if you found a simplex that has x>0
                break
        
        return x, vertices.T, num_comps
    
    __call__ = get_phase_compositions
    
    def as_dict(self):
        """ Get a output dictonary
        Utility function to get output of the 
        legacy functions in phase.py and parphase.py
        
        For the dictonary structure look at docstring of polyphase.phase 
        or polyphase.parphase
        """
        if not self.is_solved:
            raise RuntimeError('Phase diagram is not computed\n'
                               'Use .solve() before calling this method')

        outdict = {}
        outdict['config'] = []
        outdict['grid'] = self.grid
        outdict['energy'] = self.energy
        outdict['hull'] = self.hull
        outdict['thresh'] = self.thresh
        outdict['upper_hull'] = self.upper_hull
        outdict['simplices'] = self.simplices
        outdict['num_comps'] = self.num_comps
        outdict['output'] = self.df
        
        return outdict
    
    def get_kwargs(self):
        """Reproduce kwargs for legacy functions

        """
        out = {
            
            'flag_refine_simplices':self.refine_simplices,
            'flag_lift_label': self.lift_label,
            'use_weighted_delaunay': False,
            'flag_remove_collinear' : False, 
            'beta':0.0, # not used 
            'flag_make_energy_paraboloid': True if self.correction=='edge' else False, 
            'pad_energy': 2,
            'flag_lift_purecomp_energy': True if self.correction=='vertex' else False,
            'threshold_type': self.thresholding ,
            'thresh_scale':self.thresh_scale if self.thresholding=='uniform' else 1,
            'lift_grid_size':self.meshsize,
            'verbose' : self.verbose
         }
        
        return out
    
    def plotter(self):
        """A helper function for a quick visualization
        For more details on the plotting, look at `polyphase.visuals`
        """
        if self.dimension==3:
            self.renderer = TernaryPlot(self)
        elif self.dimension==4:
            self.renderer = QuaternaryPlot(self)
        else:
            raise Exception('For dimensions>4, not renderings exists')
            
        self.renderer.plot_simplices()
        plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
