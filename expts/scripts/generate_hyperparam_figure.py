import numpy as np
import polyphase as phase
import matplotlib.pyplot as plt

M = [5,5,1]
chi = [1, 0.5, 0.5]
f = lambda x : phase.flory_huggins(x, M, chi)
dxs = [100,200,300]
Deltas = [5,10,20]
fig,axs = plt.subplots(3,3,subplot_kw={'projection':'ternary'}, figsize=(8,8))
fig.subplots_adjust(wspace=0.5, hspace=0.6)
label_spacing = [1/6,3/6,5/6]
for i,dx in enumerate(dxs):
    for j, Delta in enumerate(Deltas):
        engine = phase.PHASE(f,dx,3)
        engine.compute(use_parallel=False, verbose=False,
                       thresh_scale=Delta, lift_label=True)
        ax, cbar = phase.plot_mpltern(engine.grid, engine.simplices, engine.num_comps, ax=axs[i,j])
        ax.set_tlabel('')
        ax.set_llabel('')
        ax.set_rlabel('')
        cbar.remove()
        fig.text(label_spacing[i], 0.04, '{}'.format(dx), ha='center')
        fig.text(0.04, label_spacing[j], '{}'.format(Delta), va='center', rotation='vertical')
        print('dx={}, Delta={}'.format(dx, Delta))

fig.text(0.5, 0.0, r'd$x$', ha='center')
fig.text(0.0, 0.5, r'$\Delta$', va='center', rotation='vertical')
plt.savefig('../figures/final/hyperparams.png', dpi=400, bbox_inches='tight')