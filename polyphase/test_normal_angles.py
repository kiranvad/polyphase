from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib import colors
from collections import Counter
import pdb
import polyphase as phase
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import numpy as np

class CentralDifference:
    """Compute central difference gradinet of energy
    Works only for a 3-dimensional grid or a ternary system
    
    Example:
    
    delta = np.linalg.norm(grid[:2,0] - grid[:2,1])
    cd = CentralDifference(grid, energy)    
    df = np.asarray([cd(x,y, h = delta) for x,y in grid[:-1,:].T])
    
    
    """
    def __init__(self, grid, energy):
        assert grid.shape[0]==3,'expected ternary got {}'.format(grid.shape[0])
        triang = mtri.Triangulation(grid[0,:], grid[1,:])
        self.interp_lin = mtri.LinearTriInterpolator(triang, energy)

    def __call__(self,x,y, h = 1e-3):
        """
        x,y : coordinates (float)
        h   : gridspacing (float)

        """
        f_right = self.interp_lin(x+h,y).data.squeeze()
        f_left = self.interp_lin(x-h,y).data.squeeze()
        df_dx = (f_right - f_left)/2*h
        f_right = self.interp_lin(x,y+h).data.squeeze()
        f_left = self.interp_lin(x,y-h).data.squeeze()
        df_dy = (f_right - f_left)/2*h
        
        return [df_dx, df_dy]
    
    def plot_interpolated_energy(self):
        xi, yi = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        z = self.interp_lin(xi, yi)
        fig, ax = plt.subplots()
        ax.contourf(xi, yi, z)
        plt.show()

def threecomp_gradphi(x,M,chi, beta=0):
    assert len(M)==3,'expected ternary got {}'.format(len(M))
    
    CHI = phase._utri2mat(configuration['chi'],3)

    dEdx1 = (1/M[0])*(1+np.log(x[0])) - (1/M[2])*(1+np.log(x[2])) + CHI[0,1]*x[0] +\
    CHI[0,2] - 2*CHI[0,2]*x[0] - CHI[0,2]*x[1] - CHI[1,2]*x[1] + beta*((1/x[2]**2) - (1/x[0]**2))
    
    dEdx2 = (1/M[1])*(1+np.log(x[1])) - (1/M[2])*(1+np.log(x[2])) + CHI[0,1]*x[0] -\
    CHI[0,2]*x[0] + CHI[1,2] - 2*CHI[1,2]*x[1]  - CHI[1,2]*x[0]+ beta*((1/x[2]**2) - (1/x[1]**2))
    
    return [dEdx1, dEdx2]

class TestAngles:
    def __init__(self, out, phase=2, **kwargs):
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
    
    def get_angles(self,use_findiff=True, **kwargs):
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

        if use_findiff:
            delta = np.linalg.norm(self.grid[:2,0] - self.grid[:2,1])
            cd = CentralDifference(self.grid, self.energy)    
            df = np.asarray([cd(x,y, h = delta) for x,y in self.grid[:-1,:].T])
        
        thetas = {}
        gradients = {}
        for i, (v,e) in enumerate(zip(self.vertices,
                                      self.energy[self.rnd_simplex])):
            x1,x2,_ = v
            
            if not use_findiff:
                configuration = self.out_['config']
                dx,dy = threecomp_gradphi(v,configuration['M'], configuration['chi'], beta=self.beta)
            else:
                dx = df[self.rnd_simplex[i],0]
                dy = df[self.rnd_simplex[i],1]
                
            ru = [1,0,dx]
            rv = [0,1,dy]
            uru = ru/np.linalg.norm(ru)
            urv = rv/np.linalg.norm(rv)
            normal_p = np.cross(ru, rv)
            
            angle = self._angle_between_vectors(self.facet_normal, normal_p)
            thetas.update({i:(self.rnd_simplex[i], normal_p, angle)})
            gradients.update({i:(dx,dy)})
            
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
        
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        boundary_points= np.asarray([phase.is_boundary_point(x) for x in self.grid.T])
        
        # plot energy surface
        if 1 in required:
            ps = ax.plot_trisurf(self.grid[0,~boundary_points], self.grid[1,~boundary_points], 
                                 self.energy[~boundary_points],
                                 linewidth=0.01, antialiased=True)
            ps.set_alpha(0.5)
            ax.set_xlabel('Polymer')
            ax.set_ylabel('Small molecule')
            ax.set_zlabel('Energy')
            ax.view_init(elev=16, azim=54)
        
        # plot simplex as a triangle
        if 2 in required:
            poly = Poly3DCollection(self.parametric_points,  alpha=1.0, lw=1.0, 
                                    facecolors=['tab:red'], edgecolors=['k'])
            ax.add_collection3d(poly)


                
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
                  length=0.1, normalize=True, color='sienna' )
        
        # plot phase diagram in 2D
        labels = self.X.loc['label',:].to_numpy()
        phase_colors =['r','g','b']
        if 3 in required:
            for i in [1,2,3]:
                criteria = np.logical_and(labels==i, ~boundary_points)
                ax.scatter(self.grid[0,criteria], self.grid[1,criteria], zs=-0.5, zdir='z',
                           c=phase_colors[int(i-1)])



        return fig
    
