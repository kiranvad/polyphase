import pdb
import pandas as pd
import numpy as np

from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as plt

import sys
if '../' not in sys.path:
    sys.path.insert(0,'../')
    
from parallel.parphase import compute
from solvers.visuals import plot_3d_phasediagram

from solvers.visuals import plot_mpltern, plot_lifted_label_ternary

""" configure your material system """
dimensions = 3
M = np.array([100,5,1])
chi = [0.35,0.75,0.88]
configuration = {'M': M, 'chi':chi}
dx = 400
thresh = 20
output, simplices, grid, num_comps = compute(dimensions, configuration, dx,
                                             flag_refine_simplices=True, thresh=thresh, flag_lift_label=True)

""" Post-processing """
ax, cbar = plot_mpltern(grid, simplices, num_comps)
plt.savefig('../figures/meshsize/{}.png'.format(dx),dpi=500, bbox_inches='tight')
plt.close()
ax, cbar = plot_lifted_label_ternary(output)
plt.savefig('../figures/meshsize/pointcloud_{}.png'.format(dx),dpi=500, bbox_inches='tight')
plt.close()




