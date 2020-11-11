# Phase Modelling Using Convex Envelope Method

To install, run the following commands (with git and python pip installed)
```bash
git clone https://github.com/kiranvad/polyphase.git
cd polyphase
```

Install `polyphase` as a Python package:
```python
pip install -e .
```
This will check and install polyphase and other dependencies namely : numpy, scipy, matplotlib, pandas and ray (for parallel comutation), mpltern (for ternary phase diagram visualization), plotly(for intereactive energy landscape visualization)

A sample use case is as follows:

```python
import polyphase as phase

M = [5,5,1]
chi = [1, 0.5, 0.5]
f = lambda x : phase.flory_huggins(x, M, chi)
engine = phase.PHASE(f,400,3)
engine.compute(use_parallel=False, verbose=True, lift_label=True)
```
This generates phase diagrams of the confuguration mentioned.
Few useful visualization tools are also provided. For example, phase diagram generated above can be visualized using:
```python
import matplotlib.pyplot as plt

phase.plot_mpltern(engine.grid, engine.simplices, engine.num_comps)
plt.show()
```

If you would like to run a multi-node parallel high-throughput computation use the serial version instead by replacing the compute line as below:
```python
engine.compute(use_parallel=True, verbose=True, lift_label=True)
```

To run a simple example in jupyter use the following [example](/notebooks/example.ipynb)
