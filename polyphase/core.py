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
            
        Methods:
        --------
            as_dict      :  Return the attributes of the class as a dictonary
            get_kwargs   :  Return settings of the compute method as kwargs for the private functions in _phase.py
            compute      :  Compute a phase diagram
            __call__     :  Once the phase diagram is solved using .compute(), returns the phase splitting 
                            ratios given a composition array
            plot         :  Visualize the phase diagram of 3 and 4 components
            
                                                 
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
            coplanar     :  a list of boolean values one for each simplex (True- coplanar, False- not, None- Not computed)
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
            use_parallel        : (bool) whether to use a parallel computation (default, False)
            
            verbose             : (bool) whether to print more information as the computation progresses
            
            correction          : (int) Type of energy correction to be performed:
                                        1 -- Correct the energy at the boundary of the hyperplane
                                        dimension -- No energy correction is performed
                                        anyother -- Correct compositions with that many zeros
                    
            pad_energy          : (float) Scale the energy where the correction is request to maximum energy 
                                          time the 'pad_energy'     
                                          
            lift_label          : (bool) whether to interpolate the label of a simplex into points 
                                        inside it (default, True)
                                        
            lower_hull_method   : (string or None) Method to use to obtain a lower hull from the convex hull (default : None)
                                       whether to remove simplices that are on the "upper convex hull".
                                       Note that the Gibbs criteria uses the lower convex hull, 
                                       thus it is recommended 
                                       1. None -- Defaults to using the approach where we simply remove the simplices that
                                          connect boundaries of a given energy landscape
                                       2. 'point_at_infinity' -- Computes the lower convex hull by adding an imaginary point at 
                                          the infinity height of the landscape. 
                                       3. 'negative_znorm' -- Simply assumes that the upper hull consists of simplices whose 
                                          normal in the height direction is positive.   
                                              
            thresh_scale        : (float) scaling to be used for the edge length of the reference 
                                         in thresholding
        
        NOTES: 
        ------
        In parallel mode, energy correction is not used, the lower convex hull is computed using the point at 
        infinity method instead.
        
        """
        
        self.use_parallel = kwargs.get('use_parallel', False)
        self.verbose = kwargs.get('verbose', False)
        self.correction = kwargs.get('correction', self.dimension)
        self.pad_energy = kwargs.get('pad_energy', 2)
        self.lift_label = kwargs.get('lift_label',True)
        self.refine_simplices = kwargs.get('refine_simplices',True)
        self.lower_hull_method = kwargs.get('lower_hull_method', None)
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
            'lower_hull_method' : self.lower_hull_method,
            'flag_lift_label': self.lift_label,
            'flag_remove_collinear' : False, 
            'pad_energy': self.pad_energy,
            'energy_correction': self.correction,
            'thresh_scale':self.thresh_scale, 
            'lift_grid_size':self.meshsize,
            'verbose' : self.verbose
         }
        
        return out
    
    def plot(self):
        """A helper function for a quick visualization
        For more details on the plotting, look at `polyphase.visuals`
        """
        
        if self.dimension==3:
            renderer = TernaryPlot(self)
        elif self.dimension==4:
            renderer = QuaternaryPlot(self)
        else:
            raise Exception('For dimensions>4, no renderings exists')
            
        renderer.show()
        plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
