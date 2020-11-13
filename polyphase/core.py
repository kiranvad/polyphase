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

MIN_POINT_PRECISION = 1e-8
                     
class PHASE:
    def __init__(self,energy_func, meshsize,dimension):
        if not callable(energy_func):
            raise ValueError('Vairable energy needs to be a function such as utils.flory_huggins')
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
            use_parallel   : (bool) whether to use a parallel computation (default, False)
        """
        
        self.use_parallel = kwargs.get('use_parallel', False)
        self.verbose = kwargs.get('verbose', False)
        self.correction = kwargs.get('correction', 'edge')
        self.lift_label = kwargs.get('lift_label',False)
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

            #x = np.linalg.solve(A, b)
            
            lb = np.zeros(self.dimension)
            ub = np.ones(self.dimension)
            res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=0)
            x = res.x
            if not (x<0).any():
                # STOP if you found a simplex that has x>0
                break
        
        return x, vertices, num_comps
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
