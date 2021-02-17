from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib import colors
from collections import Counter
import pdb
import polyphase as phase
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import mpltern
import numpy as np
from scipy.spatial.distance import pdist, cdist
from itertools import combinations
from .helpers import inpolyhedron
from .visuals import _set_axislabels_mpltern

class CentralDifference:
    """Compute central difference gradinet of energy
    Works only for a 3-dimensional grid or a ternary system
    given the energy function
    """
    def __init__(self, grid, energy):
        if callable(energy):
            self.func = energy
        else:
            raise RuntimeError

    def __call__(self,phi, h = 1e-3):
        """
        x,y : coordinates (float)
        h   : gridspacing (float)

        """
        p1,p2,p3 = phi
        f_right = self.func([p1+h,p2,1-p1-h-p2])
        f_left = self.func([p1-h,p2,1-p1+h-p2])
        df_dx = (f_right - f_left)/(2*h)
        
        f_right = self.func([p1,p2+h,1-p1-h-p2])
        f_left = self.func([p1,p2-h,1-p1+h-p2])
        df_dy = (f_right - f_left)/(2*h)
        
        return [df_dx, df_dy]

class base:
    def __init__(self,engine, phase=2,simplex_id= None):
        self.engine = engine
        self.grid = self.engine.grid
        self.num_comps = self.engine.num_comps
        self.simplices = self.engine.simplices
        self.energy = self.engine.energy
        self.X = self.engine.df
        self.phase = phase
        self.coplanar = self.engine.coplanar
        
        if simplex_id is None:
            self.get_random_simplex()
        else:
            self.set_simplex_data(simplex_id)
        
        v = self.vertices[:,:-1]
        inside = inpolyhedron(v, self.grid[:-1,:].T)
        self.interval = self.grid[:,inside]
        
    def get_random_simplex(self):
        phase_simplices_ids = np.where(np.asarray(self.num_comps)==self.phase)[0]
        self.rnd_simplex_indx = np.random.choice(phase_simplices_ids,1)
        self.set_simplex_data(self.rnd_simplex_indx)
    
    def set_simplex_data(self, simplex_id):
        self.rnd_simplex_indx = simplex_id
        self.rnd_simplex = self.simplices[simplex_id].squeeze()
        self.vertices = self.X.iloc[:3,self.rnd_simplex].to_numpy().T
        self.parametric_points = np.hstack((self.vertices[:,:2],
                                            self.energy[self.rnd_simplex].reshape(-1,1))).tolist()
        if not self.coplanar[simplex_id]:
            raise RuntimeError('The chosen simplex with vertices {} is flat.'
                               'The tests cannot be performed.'.format(self.vertices))
    
    def base_visualize(self):
        """ Visualize the test case base function
        
        Plot the usual suspects required: simplex

        """
        
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        self.boundary_points= np.asarray([phase.is_boundary_point(x) for x in self.grid.T])

        poly = Poly3DCollection(self.parametric_points,  alpha=1.0, lw=1.0, 
                                facecolors=['tab:gray'], edgecolors=['k'])
        ax.add_collection3d(poly)
        ax.set_label('Energy landscape')
        ax.set_xlabel('Polymer')
        ax.set_ylabel('Small molecule')
        ax.set_zlabel('Energy')
        ax.view_init(elev=16, azim=60)
        
        return fig, ax        
        
        
class TestAngles(base):
    def __init__(self, engine,phase=2,simplex_id=None, **kwargs):
        """ Perform a test to compute angles of tangent planes at vertices to convex combination of points
        Test takes the out from polyphase.compute or polyphase.serialcompute and the same kwargs
        """
        super().__init__(engine,phase=phase,simplex_id=simplex_id,**kwargs)

    def get_angles(self,gradient,**kwargs):
        """Compute angles between tangent planes at the simplex vertices and facet normal
        
        Facet normal can be compute by generating a plane equation or using the hull facet equations
        from `out_[hull].equations`. 
        
        use_findiff trigger the gradient computation using central difference of an interpolated energy
        see `class::CentralDifference` for more details
         
         
        returns dictonary of dictonaries with the following keys:
        'facet_normal' : facet normal of simplex from the convexhull
        'thetas'       : dictonary with vertices named in numeric keys (0,1,2) with each numeric key
                         containing the tuple (simplex id, normal to the tangent plane, angle with facet normal)
                         
        'gradients'    : dictonary with vertices named in numeric keys (0,1,2) with each numeric key
                         containing the tuple (df_dx, df_dy)
        """

        all_facet_equations = self.engine.hull.equations[~self.engine.upper_hull]
        facet_equation = all_facet_equations[self.rnd_simplex_indx].squeeze()
        self.facet_normal = facet_equation[:-1]
            
        thetas = {}
        gradients = {}
        for i, (v,e) in enumerate(zip(self.vertices,self.energy[self.rnd_simplex])):
            normal_p, dx, dy = self._get_normal(v, gradient)
            angle = self._angle_between_vectors(self.facet_normal, normal_p)
            dot_product = np.abs(np.clip(np.dot(self.facet_normal, normal_p), -1.0, 1.0))
            
            # tuple of simplex, tangent normal, andle and dot product with facet normal                     
            thetas.update({i:(self.rnd_simplex[i], normal_p, angle, dot_product)})
            # tuple of gradients along phi_1, phi_2 and the normal of the tangent plane
            gradients.update({i:(dx,dy, normal_p)}) 

        outdict = {'facet_normal': self.facet_normal, 'thetas':thetas, 'gradients':gradients}  
        
        self._angles_outdict = outdict
        
        return outdict
    
    def _get_normal(self, v, gradient):
        x1,x2,_ = v

        dx,dy = gradient(v)

        ru = [1,0,dx]
        rv = [0,1,dy]
        uru = ru/np.linalg.norm(ru)
        urv = rv/np.linalg.norm(rv)
        normal = np.cross(ru, rv)
        
        return normal, dx, dy
    
    def _angle_between_vectors(self, v,w):
        """Compute angle between two n-dimensional Euclidean vectors
        
        from : https://stackoverflow.com/a/13849249
        
        """
        v = v / np.linalg.norm(v)
        w = w / np.linalg.norm(w)
        
        return np.degrees(np.arccos(np.clip(np.dot(v, w), -1.0, 1.0)))

    def visualize(self, required=[1,2]):
        """ Visualize the test case
        
        By default plots: 
            1. Energy landscape
            2. Simplex selected
            3. phase diagram in \phi_{1,2}
            4. Tangent plane generators at the vertices
            -. Facet normal of the random simplex selected
            -. Normal vectors at the simplices vertices derived in `get_angles` 
            (-. means always plotted)
            
        To plot only selection of the above, pass a list with respective integers in the argument 'required'    
        
        NOTE: This function won't work without calling get_angles() first

        """
        fig, ax = self.base_visualize()
        
        # plot energy surface
        if 1 in required:
            ps = ax.plot_trisurf(self.grid[0,~self.boundary_points], self.grid[1,~self.boundary_points], 
                                 self.energy[~self.boundary_points],
                                 linewidth=0.01, antialiased=True)
            ps.set_alpha(0.5)
    
        for i, pp in enumerate(self.parametric_points):
            uv = self._angles_outdict['thetas'][i][1]
            if 4 in required:
                ax.quiver(v[0], v[1], e, uru[0],uru[1],uru[2], length=0.1, normalize=True, color='k')
                ax.quiver(v[0], v[1], e, urv[0],urv[1],urv[2], length=0.1, normalize=True, color='purple')
            ax.quiver(pp[0], pp[1], pp[2], uv[0], uv[1], uv[2], length=0.1, normalize=True, color='tab:red', 
                      label='Tangent normal' if i==0 else '')
                
        facet_normal = self._angles_outdict['facet_normal']
        rnd_simplex_centroid = np.mean(self.parametric_points, axis=0)
        ax.quiver(rnd_simplex_centroid[0], rnd_simplex_centroid[1], rnd_simplex_centroid[2],
                  facet_normal[0], facet_normal[1], facet_normal[2], 
                  length=0.1, normalize=True, color='k', label='Facet normal')
        
        # plot phase diagram in 2D
        labels = self.X.loc['label',:].to_numpy()
        phase_colors =['r','g','b']
        if 3 in required:
            for i in [1,2,3]:
                criteria = np.logical_and(labels==i, ~boundary_points)
                ax.scatter(self.grid[0,criteria], self.grid[1,criteria], zs=-0.5, zdir='z',
                           c=phase_colors[int(i-1)])
        
        return fig, ax
    

class TestEpiGraph(base):
    def __init__(self, engine,phase=2,simplex_id=None, **kwargs):
        super().__init__(engine,phase=phase,simplex_id=simplex_id,**kwargs)
        self.f = engine.energy_func
        
    def is_epigraph(self):
        self.ABC = self._get_plane_equation()
        results = []
        for point in self.interval.T:
            f_actual = self.f(point)
            f_convex = self._get_convex_energy(point)
            
            is_lower_envelope = np.greater_equal(f_actual,f_convex)
            if not is_lower_envelope:
                if np.isclose(f_actual, f_convex):
                    is_lower_envelope = True
                else:
                    print(point, f_actual, f_convex)
                
            results.append(is_lower_envelope)
        
        return np.asarray(results).all()

    def _get_plane_equation(self):
        A = np.asarray(self.parametric_points)
        b = np.array([1,1,1])
        x = np.linalg.solve(A, b)
        
        return x
    
    def _get_convex_energy(self, point):
        """Return energy approximated by the convex hull 
        at a given composition
        
        """
        return (1/self.ABC[2])*(1-self.ABC[0]*point[0]-self.ABC[1]*point[1])
    
    def visualize(self):
        """ Visualize the epigraph and simplex 
        
        wrapper function for base_visualize that additionally plots the point evaluations
        of the energy function given in self.f
        """
        
        fig, ax = self.base_visualize()
        
        f_actual = [self.f(p) for p in self.interval.T]
        f_convex = [self._get_convex_energy(p) for p in self.interval.T]
        ax.plot_trisurf(self.interval[0,:], self.interval[1,:], f_actual,
                                 linewidth=0.01, antialiased=True, fc = 'tab:blue')
        #ax.scatter(self.interval[0,:], self.interval[1,:], f_actual, label='Energy function')
        #ax.scatter(self.interval[0,:], self.interval[1,:], f_convex)
        ax.set_zlim(min(f_convex), max(f_actual))
        
        return fig, ax
    

class TestPhaseSplits(base):
    """Test if a simplex splits the points according to its labels
    
    This test performs whether a simplex with a phase label splits them into
    the correct amount of significant phases
    
    Inputs:
    -------
        out  : a polyphase.PHASE instance solved for phase diagram
        
    Methods:
    --------
        is_correct_phasesplit : determines whether phase splits obtained are in accordance
                                with simplex labels (works only for two and three phases)
        run                   : tests the phase splits matching for any selected simplex
    """
    def __init__(self, engine,phase=2, simplex_id=None, threshold=0.1):
        super().__init__(engine,phase=phase,simplex_id=simplex_id)
        self.engine = engine
        self.threshold = threshold
        indx = list(combinations(range(3),2))
        self.min_edge_verts = indx[np.argmin(pdist(self.vertices))]
        
    def ratios_are_close(self, splits):
        """Computes if split ratios are close
        
        This method can be called to identify if the ratio of splits 
        for two compositonatlly similar stable phases are close
        """

        a,b = splits[np.asarray(self.min_edge_verts)]
        return np.isclose(a,b)
    
    def is_correct_phasesplit(self,splits):
        """Check if the phase splits match the phase label of the simplex
        
        This function assumes that two-phase simplex will have phase split
        ratios such that atleast one value is lesser than a threshold
        
        Inputs:
        -------
            splits.   : splitting ratios of a point obtained by calling `self.engine`
            threshold : Threshold value to consider any phase split ratio to be insignificant
                        (default, 0.01 or 1%)
                        
        Returns:
        --------
            A boolean value to suggest whether the simplex label and phase split based label matched
     
        """
        splits = np.asarray(splits)
        if self.phase==2:
            matches = splits<self.threshold
            #matches = np.append(matches, self.ratios_are_close(splits))
            return matches.any()
        if self.phase==3:
            matches = splits>self.threshold
            return matches.all()    
    
    def check_centroid(self):
        """Check if simplex can split the centroid as labelled
        """
        points = np.asarray(self.parametric_points)
        self.centroid_ = np.sum(points, axis=0)/3
        x, self.equilibrium_phases_, _ = self.engine(self.centroid_)
        is_match = self.is_correct_phasesplit(x)
        if not is_match and self.phase==2:
            decision,_,_ = self._min_edge_lengths_equal()
        else:
            decision = is_match
            
        self.centroid_splits_ = x
        
        return decision
    
    def _min_edge_lengths_equal(self):
        
        v = self.vertices[self.min_edge_verts,:-1]
        a = np.linalg.norm(v[0,:]-self.centroid_[:-1])
        b = np.linalg.norm(v[1,:]-self.centroid_[:-1])
        
        return np.isclose(a,b, atol=1e-2), a,b
    
    
    def run(self):
        """Check if a simplex phase splits all points inside correctlty
        
        Attributes:
        -----------
            results              : boolean list indexed by points inside the simplex if 
                                   the simplex phase splits the point according to its label
            non_matching_splits_ : splits of points that did not match the label of simplex
            matching_splits_     : splits of points that matched the simplex label
        """
        results = []
        self.non_matching_splits_ = []
        self.matching_splits_ = []
        for p in self.interval.T:
            x, _, _ = self.engine(p)
            is_match = self.is_correct_phasesplit(x)
            if not is_match:
                self.non_matching_splits_.append(x)
            else:
                self.matching_splits_.append(x)
                
            results.append(is_match)
        
        self.results = np.asarray(results)
        assert len(self.results)>0, 'No result is stored'

        self.total_match = np.sum(self.results)
        
        return self.results.all()
    
    def visualize(self, ax=None, show=True):
        """Visualize points where labels didn't match
        
        Can only call this method after .run()
        """
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection':'ternary'})
        else:
            fig = plt.gcf()
            
        ax.fill(self.vertices[:,2], self.vertices[:,0], self.vertices[:,1],
                alpha=0.2, facecolor='gray', edgecolor='none')
        ax.scatter(self.interval.T[~self.results,2], self.interval.T[~self.results,0],
                   self.interval.T[~self.results,1], color='tab:red')
        
        _set_axislabels_mpltern(ax)
        
        if show:
            plt.show()
            return
        else:
            return fig, ax
    
    def visualize_centroid(self, ax=None, show=True):
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        patch = Polygon(self.vertices[:,:-1], True)  
        p = PatchCollection([patch], alpha=0.4, fc='gray')
        ax.add_collection(p)
        ax.scatter(self.vertices[:,0], self.vertices[:,1], color='k')
        ax.scatter(self.centroid_[0], self.centroid_[1], color='tab:red')
        if show:
            plt.show()
            return
        else:
            return fig, ax
    
    

    
    
    
    