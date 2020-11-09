from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib import colors
from collections import Counter
import pdb
import polyphase as phase
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import numpy as np

from .helpers import inpolyhedron

class base:
    def __init__(self,out,*, phase=2,**kwargs):
        self.grid = out['grid']
        self.num_comps = out['num_comps']
        self.simplices = out['simplices']
        self.energy = out['energy']
        self.X = out['output']
        self.out_ = out
        
        self.phase = phase
        self.beta = kwargs['beta']
        self.__dict__.update(kwargs)
        self.get_random_simplex()
        
    def get_random_simplex(self):
        phase_simplices_ids = np.where(np.asarray(self.num_comps)==self.phase)[0]
        self.rnd_simplex_indx = np.random.choice(phase_simplices_ids,1)
        self.rnd_simplex = self.simplices[self.rnd_simplex_indx].squeeze()
        self.vertices = self.X.iloc[:3,self.rnd_simplex].to_numpy().T
        self.parametric_points = np.hstack((self.vertices[:,:2],
                                            self.energy[self.rnd_simplex].reshape(-1,1))).tolist()

    def base_visualize(self,**kwargs):
        """ Visualize the test case base function
        
        Plot the usual suspects required: simplex (with required=2, and its facet normal)

        """
        
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        self.boundary_points= np.asarray([phase.is_boundary_point(x) for x in self.grid.T])

        poly = Poly3DCollection(self.parametric_points,  alpha=1.0, lw=1.0, 
                                facecolors=['tab:gray'], edgecolors=['k'])
        ax.add_collection3d(poly)

        ax.set_xlabel('Polymer')
        ax.set_ylabel('Small molecule')
        ax.set_zlabel('Energy')
        ax.view_init(elev=16, azim=54)
        
        return fig, ax        
        
        
class TestAngles(base):
    def __init__(self, out, *,phase=2, **kwargs):
        """ Perform a test to compute angles of tangent planes at vertices to convex combination of points
        Test takes the out from polyphase.compute or polyphase.serialcompute and the same kwargs
        
        Example:
        otu = polyphase.compute(**)
        test = TestAngles(out,phase=1,**kwargs)
        test_out = test.get_angles(use_findiff=True)
        for key, value in test_out['thetas'].items():
            print('Angle at vertex {} is {:.2f} degrees'.format(key, value[2]))

        fig = test.visualize()
        plt.show()
        
        
        """
        super().__init__(out,phase=phase,**kwargs)

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

        all_facet_equations = self.out_['hull'].equations[~self.out_['upper_hull']]
        facet_equation = all_facet_equations[self.rnd_simplex_indx].squeeze()
        self.facet_normal = facet_equation[:-1]
            
        thetas = {}
        gradients = {}
        for i, (v,e) in enumerate(zip(self.vertices,
                                      self.energy[self.rnd_simplex])):
            x1,x2,_ = v
            
            dx,dy = gradient(v)

            ru = [1,0,dx]
            rv = [0,1,dy]
            uru = ru/np.linalg.norm(ru)
            urv = rv/np.linalg.norm(rv)
            normal_p = np.cross(ru, rv)
            
            angle = self._angle_between_vectors(self.facet_normal, normal_p)
            thetas.update({i:(self.rnd_simplex[i], normal_p, angle)})
            gradients.update({i:(dx,dy, normal_p)}) # tuple of gradients along phi_1, phi_2 and the normal of the tangent plane
            
        outdict = {'facet_normal': self.facet_normal, 'thetas':thetas, 'gradients':gradients}  
        
        self._angles_outdict = outdict
        
        return outdict
    
    def _angle_between_vectors(self, v,w):
        """Compute angle between two n-dimensional Euclidean vectors
        
        from : https://stackoverflow.com/a/13849249
        
        """
        v = v / np.linalg.norm(v)
        w = w / np.linalg.norm(w)
        
        return np.degrees(np.arccos(np.clip(np.dot(v, w), -1.0, 1.0)))

    def visualize(self, required=[1,2,3]):
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
            ax.quiver(pp[0], pp[1], pp[2], uv[0], uv[1], uv[2], length=0.1, normalize=True, color='tab:red' )
                
        facet_normal = self._angles_outdict['facet_normal']
        rnd_simplex_centroid = np.mean(self.parametric_points, axis=0)
        ax.quiver(rnd_simplex_centroid[0], rnd_simplex_centroid[1], rnd_simplex_centroid[2],
                  facet_normal[0], facet_normal[1], facet_normal[2], 
                  length=0.1, normalize=True, color='k' )
        
        # plot phase diagram in 2D
        labels = self.X.loc['label',:].to_numpy()
        phase_colors =['r','g','b']
        if 3 in required:
            for i in [1,2,3]:
                criteria = np.logical_and(labels==i, ~boundary_points)
                ax.scatter(self.grid[0,criteria], self.grid[1,criteria], zs=-0.5, zdir='z',
                           c=phase_colors[int(i-1)])
        
        return fig
    

class TestEpiGraph(base):
    def __init__(self, out,energy_func, *,phase=2, **kwargs):
        super().__init__(out,phase=phase,**kwargs)
        v = self.vertices[:,:-1]
        inside = inpolyhedron(v, self.grid[:-1,:].T)
        self.interval = self.grid[:,inside]
        self.f = energy_func
        
    def is_epigraph(self):
        self.ABC = self._get_plane_equation()
        results = []
        for point in self.interval.T:
            f_actual = self.f(point)
            f_convex = self._get_convex_energy(point)
            
            is_lower_envelope = np.greater_equal(f_actual,f_convex)
            if not is_lower_envelope:
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
        of the energy function given in 'energy_func' (self.f)
        """
        
        fig, ax = self.base_visualize()
        
        f_actual = [self.f(p) for p in self.interval.T]
        f_convex = [self._get_convex_energy(p) for p in self.interval.T]
        
        ax.scatter(self.interval[0,:], self.interval[1,:], f_actual, label='Energy function')
        ax.scatter(self.interval[0,:], self.interval[1,:], f_convex, label='Convex approx')
        ax.set_zlim(min(f_convex), max(f_actual))
        
        return fig, ax
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    