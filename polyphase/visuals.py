""" Bunch of visualization tools that aid analysis """
import matplotlib.pyplot as plt

import pdb
import numpy as np
import mpltern
from matplotlib.cm import ScalarMappable
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ._phase import is_boundary_point

def _set_axislabels_mpltern(ax):
    """ 
    Sets axis labels for phase plots using mpltern 
    in the order of solvent (index 2), polymer (index 0), non-solvent (index 1)
    """
    ax.set_tlabel(r'$\phi_2$', fontsize=15)
    ax.set_llabel(r'$\phi_1$', fontsize=15)
    ax.set_rlabel(r'$\phi_3$', fontsize=15)
    ax.taxis.set_label_position('tick1')
    ax.laxis.set_label_position('tick1')
    ax.raxis.set_label_position('tick1')   

def plot_energy_landscape(outdict,mode='full', ax = None):
    """ Plots a convex hull of a energy landscape 
    
    parameters:
    -----------
        outdict     :  polyphase.PHASE.as_dict()
    
    This function takes an optional argument in mode which can be used to 
    visualize the just the convex hull (mode='convex_hull') approximation instead
    By default it plots the triangulated energy landscape (mode='full')
    This function plots the energy landscape with a thin boundary 
    cut around the two phase composotions
    """
    grid = outdict['grid']
    assert grid.shape[0]==3, 'Expected a ternary system but got {}'.format(grid.shape[0])

    boundary_points= np.asarray([is_boundary_point(x) for x in grid.T])
    energy = outdict['energy']
 
    if ax is None:
        fig = plt.figure(figsize=(4*1.6, 4))
        ax = fig.add_subplot(projection='3d')
    else:
        fig = plt.gcf()
        
    if mode=='full':    
        ax.plot_trisurf(grid[0,~boundary_points], grid[1,~boundary_points], 
                        energy[~boundary_points], linewidth=0.01, antialiased=True)
    elif mode=='convex_hull':
        ax.plot_trisurf(grid[0,:], grid[1,:], 
                        energy, triangles=outdict['simplices'], 
                        linewidth=0.01, antialiased=True)
    ax.set_xlabel('Polymer')
    ax.set_ylabel('Small molecule')
    ax.set_zlabel('Energy')
    ax.set_title('Energy landscape', pad=42)
    
    return ax, fig    
    
def plain_phase_diagram(df, ax = None):
    """ 
    Plot phase diagrams as points without any labels or stuff
    Used as a data point for dimensionality reduction and clustering
    
    parameters:
    -----------
        df     :  polyphase.PHASE.df (after calling .compute(), 
                  you should have access to the attribute .df if run with 'lift_label'=True)   

    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection':'ternary'})
    else:
        fig = plt.gcf()
    
    phase_colors =['w','r','g','b']
    for i, p in df.T.groupby('label'):
        ax.scatter(p['Phi_3'], p['Phi_1'], p['Phi_2'], c=phase_colors[int(i)])
        
    plt.axis('off')
    
    return ax     
    
class TernaryPlot:
    def __init__(self,engine):
        """Plot 3-component system phase diagram in a ternary plot
        
        Inputs:
        =======
            engine    :  `polyphase.PHASE` class after .compute(*args,**kwargs) is called
            
        Methods:
        ========
        
        """
        self.engine = engine
        
    def plot_simplices(self,ax=None,label=True):
        """A phase diagram with simplices glued together with phase colorcoded 
        Input:
        --------
            ax            :  matplotlib.pyplot.Axis object
            label         :  (boolean, True) Whether to add labels to the plot

        """

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection':'ternary'})
        else:
            fig = plt.gcf()
            
        self._check_ternary_projection(ax)
        
        phase_colors =['tab:red','tab:olive','tab:cyan']
        cmap = colors.ListedColormap(phase_colors)
        for l,s in zip(self.engine.num_comps, self.engine.simplices):
            simplex_points = np.asarray([self.engine.grid[:,x] for x in s])
            ax.fill(simplex_points[:,2], simplex_points[:,0], simplex_points[:,1], facecolor=phase_colors[int(l-1)])
        if label:
            _set_axislabels_mpltern(ax)
        boundaries = np.linspace(1,4,4)
        norm = colors.BoundaryNorm(boundaries, cmap.N)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        cax = ax.inset_axes([1.05, 0.1, 0.05, 0.9], transform=ax.transAxes)
        cbar = fig.colorbar(mappable,shrink=0.5, aspect=5, ticks=[1.5,2.5,3.5],cax=cax)
        cbar.ax.set_yticklabels(['1-Phase', '2-Phase', '3-Phase'])

        return ax, cbar    
    
    def plot_points(self,ax = None, label=True):
        """ A point cloud phase diagram from the lifted simplices 

        Input:
        --------
            ax            :  matplotlib.pyplot.Axis object
            label         :  (boolean, True) Whether to add labels to the plot
        """
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection':'ternary'})
        else:
            fig = plt.gcf()

        self._check_ternary_projection(ax)
        phase_colors =['w','r','g','b']
        cmap = colors.ListedColormap(phase_colors[1:])
        df = self.engine.df.T
        for i, p in df.groupby('label'):
            ax.scatter(p['Phi_3'], p['Phi_1'], p['Phi_2'], c=phase_colors[int(i)])
        if label:
            _set_axislabels_mpltern(ax)

        boundaries = np.linspace(1,4,4)
        norm = colors.BoundaryNorm(boundaries, cmap.N)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(mappable,shrink=0.5, aspect=5, ticks=[1.5,2.5,3.5],ax=ax)
        cbar.ax.set_yticklabels(['1-Phase', '2-Phase', '3-Phase'])

        return ax, cbar
    
    def show(self,*args,**kwargs):
        self.plot_simplices(*args,**kwargs)
    
    def _check_ternary_projection(self,ax):
        if not ax.name=='ternary':
            raise Exception('Axis needs to be a ternary projection')
             
class QuaternaryPlot:
    def __init__(self, engine):
        """Plot 4-component system with in a Quaternary plot
        Inputs:
        =======
            model  :  `polyphase.PHASE` class after .compute(*args,**kwargs) is called
            
        Methods:
        ========
            from4d23d             :  Convert from 4-dimensional hyperplane to 3d embedding 
                                    (conformmaly using barycentric coordinates)
            _get_convex_faces     :  Utility function to plot a 3d simplex from its vertices
            add_outline           :  Add border of a tetrahedron to the plot
            add_colored_simplices :  Add simplices (from engine.simplices) and color them based 
                                     on the phase label (engine.num_comps)
            add_scatter           :  plot points in a 3D scatter plot and color them based on the labels
            add_colorbar          :  Add a colorbar to the figure
            
            
            plot_points           :  Plot the phase diagram using the points interplotated labels from engine.df
            
            plot_simplices        :  Plot the phase diagram by gluing the simplices together from engine.simplices
            
        Attributes:
        ===========
            phase_colors     : colors used (and indexed) for each phase label 
            vertices         : Vertices of the tetrahedron outline
            threed_coords    : Three dimensional embedding coordinates. 
                               Avalaible only if the `.plot_points(*args,**kwargs)` method is called
                               
        Examples:
        =========
        1. Plot using simplices
            >>> qtplot = polyphase.QuaternaryPlot(engine)
            >>> [fig, axs, cbar] = qtplot.plot_simplices(sliceat=0.5)
            >>> fig.suptitle('Sliced at z={:.2f}'.format(t))
            >>> plt.show()
            
        2. Plot using points
            >>> qtplot = polyphase.QuaternaryPlot(engine)
            >>> [fig, axs, cbar] = qtplot.plot_points(sliceat=0.5)
            >>> fig.suptitle('Sliced at z={:.2f}'.format(t))
            >>> plt.show()    
        
            
        """
        if engine.is_solved:
            self.engine = engine
        else:
            raise RuntimeError('Requires the `polyphase.PHASE` class to be solved.')
            
        assert self.engine.dimension==4, 'This functions works only for dimension 4 but {} PHASE class is provided'.format(engine.dimension)
        
        self.vertices = np.array([[0, 0, 0], 
                                  [1, 0, 0], 
                                  [1/2,np.sqrt(3)/2,0],
                                  [1/2,np.sqrt(3)/6,np.sqrt(6)/3]]
                                )
        self.phase_colors =['tab:red','tab:olive','tab:cyan','tab:purple']
        
        self.threed_coords = []
        for i,row in self.engine.df.T.iterrows():
            self.threed_coords.append(self.from4d23d(row[:-1].to_list()))
            
        self.threed_coords = np.asarray(self.threed_coords)
        
    
    def from4d23d(self,fourd_coords):
        """Compute 3D coordinates of 4-component composition
        
        Inputs:
        =======
        fourd_coords  :  Four component composition as a list
        
        Outputs:
        ========
            [u,v,w]   : 3D coordinates
            
        """
        x,y,z,w = fourd_coords
        u = y+0.5*(z+w)
        v = np.sqrt(3)*(z/2 + w/6) 
        w = np.sqrt(6)*(w/3)

        return [u,v,w]
    
    def _get_convex_faces(self,v):
        """Return set of faces of tetrahedron simplex
        Inputs:
        =======
            v.  :  Vertices of the simplex
            
        Outputs:
        ========
            verts  :  list of face vertices
            
        """
        verts = [ [v[0],v[1],v[3]], [v[1],v[2],v[3]],\
                 [v[0],v[2],v[3]], [v[0],v[1],v[2]]]
        return verts

    def add_outline(self,ax):
        """Add tetrahedron outline to the axis
        
        ax : a 3D matplotlib.pyplot axis
        
        """
        ax.scatter3D(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2],
                     color='black')
        emb_verts = self._get_convex_faces(self.vertices)
        ax.add_collection3d(Poly3DCollection(emb_verts, facecolors='white', 
                                             linewidths=0.5, edgecolors='grey', alpha=.05))
        labels = [r'$\phi_{1}$',r'$\phi_{2}$',r'$\phi_{3}$',r'$\phi_{4}$']
        for vi,w in zip(self.vertices,labels):
            ax.text(vi[0],vi[1],vi[2],w)
            
    def add_colored_simplices(self,ax,cluster,sliceat=0.5):  
        """
        ax      : a 3D matplotlib.pyplot axis
        sliceat : (float, 0.5) Where to slice the tetraehdron in z-direction
        
        """
        flag_targets = np.asarray(self.engine.num_comps)==cluster
        
        for i,simplex in zip(np.asarray(self.engine.num_comps)[flag_targets],self.engine.simplices[flag_targets]):
            simplex_vertices = [self.engine.grid[:,x] for x in simplex]
            v = np.asarray([self.from4d23d(vertex) for vertex in simplex_vertices])
            if np.all(np.asarray(simplex_vertices)[:,3]<sliceat):
                verts = self._get_convex_faces(v)
                ax.add_collection3d(
                    Poly3DCollection(verts,facecolors=self.phase_colors[int(i-1)],
                                     edgecolors=None)
                )
                
    def add_scatter(self,ax,cluster,sliceat=0.5):
        """
        ax      : a 3D matplotlib.pyplot axis
        sliceat : (float, 0.5) Where to slice the tetraehdron in z-direction
        
        """
        
        cluster_ids = np.where(self.engine.df.T['label']==cluster)
        slice_ids = np.where(self.threed_coords[:,2]<sliceat)
        ids = np.intersect1d(slice_ids,cluster_ids)
        ax.scatter(self.threed_coords[ids,0], self.threed_coords[ids,1],
                   self.threed_coords[ids,2], color=self.phase_colors[int(cluster-1)])
    
    def plot_points(self,sliceat=0.5):
        """
        sliceat : (float, 0.5) Where to slice the tetraehdron in z-direction
        """

        self.threed_coords = np.asarray(self.threed_coords)
        
        fig, axs = plt.subplots(2,2,subplot_kw={'projection': '3d'}, figsize=(8,8))
        axs = axs.flatten()
        for i,ax in enumerate(axs):
            self.add_outline(ax)
            self.add_scatter(ax,i+1,sliceat=sliceat)
        cbar = self.add_colorbar(fig)
        
        return [fig, axs, cbar]
    
    def plot_simplices(self, sliceat=1.0):
        """
        sliceat : (float, 0.5) Where to slice the tetraehdron in z-direction
        """
        
        fig, axs = plt.subplots(2,2,subplot_kw={'projection': '3d'}, figsize=(8,8))
        axs = axs.flatten()
        for i,ax in enumerate(axs):
            self.add_outline(ax)
            self.add_colored_simplices(ax,i+1,sliceat=sliceat)
        cbar = self.add_colorbar(fig)
        
        return [fig, axs, cbar]
            
    def add_colorbar(self,fig):
        """
        fig. : matplotlib.pyplot figure handle
        """
        
        cmap = colors.ListedColormap(self.phase_colors)        
        boundaries = np.linspace(1,5,5)
        norm = colors.BoundaryNorm(boundaries, cmap.N)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        cax = fig.add_axes([1.05, 0.25, 0.03, 0.5])
        cbar = fig.colorbar(mappable,shrink=0.5, aspect=5, ticks=[1.5,2.5,3.5,4.5],cax=cax)
        cbar.ax.set_yticklabels(['1-Phase', '2-Phase', '3-Phase','4-Phase'])
        
        return cbar  
    
    def show(self, mode='simplices'):
        fig, axs = plt.subplots(2,2,subplot_kw={'projection': '3d'}, figsize=(8,8))
        axs = axs.flatten()
        for ax,t in zip(axs,[0.025,0.25,0.5,1.0]):
            ax.set_title(r'$\phi_{4}\leq$'+'{:.2f}'.format(t), pad=0.0)
            ax._axis3don = False
            self.add_outline(ax)
            if mode=='simplices':
                for i,simplex in zip(np.asarray(self.engine.num_comps),self.engine.simplices):
                    simplex_vertices = [self.engine.grid[:,x] for x in simplex]
                    v = np.asarray([self.from4d23d(vertex) for vertex in simplex_vertices])
                    if np.all(np.asarray(simplex_vertices)[:,3]<t):
                        verts = self._get_convex_faces(v)
                        ax.add_collection3d(
                            Poly3DCollection(verts,facecolors=self.phase_colors[int(i-1)],
                                             edgecolors=None)
                        )
                        
            elif mode=='points':
                for cluster in [1,2,3,4]:
                    cluster_ids = np.where(self.engine.df.T['label']==cluster)
                    slice_ids = np.where(self.engine.df.T['Phi_4']<t)
                    ids = np.intersect1d(slice_ids,cluster_ids)
                    ax.scatter(self.threed_coords[ids,0], self.threed_coords[ids,1],
                               self.threed_coords[ids,2], color=self.phase_colors[int(cluster-1)])
                
            else:
                raise RuntimeError('Only the simplices or points mode is avaliable')
        cbar = self.add_colorbar(fig)
        
        return [fig, axs, cbar]

    
    
    
    
    
    
    
    
    
        