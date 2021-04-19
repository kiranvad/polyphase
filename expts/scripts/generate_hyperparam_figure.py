import numpy as np
import polyphase
import matplotlib.pyplot as plt

M = [5,5,1]
chi = [1, 0.5, 0.5]
f = lambda x : polyphase.flory_huggins(x, M, chi)
dxs = [50,200,300]
Deltas = [5,10,20]
fig,axs = plt.subplots(3,3,subplot_kw={'projection':'ternary'}, figsize=(8,8))
#fig.subplots_adjust(wspace=0.5, hspace=0.6)
label_spacing_x = [0.25,3/6,0.75]
label_spacing_y = [0.75,3/6,0.25]
for i,dx in enumerate(dxs):
    for j, Delta in enumerate(Deltas):
        engine = polyphase.PHASE(f,dx,3)
        engine.compute(use_parallel=False, verbose=False,
                       thresh_scale=Delta, lift_label=True)
        render = polyphase.TernaryPlot(engine)
        ax, cbar = render.plot_simplices(ax=axs[i,j])
        ax.taxis.set_tick_params(which='both', bottom=False, top=False, labelbottom=False)
        ax.laxis.set_tick_params(which='both', bottom=False, top=False, labelbottom=False)
        ax.raxis.set_tick_params(which='both', bottom=False, top=False, labelbottom=False)
        cbar.remove()
        fig.text(label_spacing_x[i], 0.04, r'd$x$={}'.format(dx), ha='center')
        fig.text(0.04, label_spacing_y[j], r'$\Delta=${}'.format(Delta), va='center', rotation='vertical')
        print('dx={}, Delta={}'.format(dx, Delta))

plt.savefig('../figures/final/hyperparams.png', dpi=400, bbox_inches='tight')