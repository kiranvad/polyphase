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
chi = [0.5, 0.5, 1]
configuration = {'M':M, 'chi':chi}
meshsize = 100
out = phase.compute(configuration,meshsize) 
```
This generates phase diagrams of the confuguration mentioned.
Few useful visualization tools are also provided. For example, phase diagram generated above can be visualized using:
```python
import matplotlib.pyplot as plt

grid = out['grid']
num_comps = out['num_comps']
simplices = out['simplices']
phase.plot_mpltern(grid, simplices, num_comps)
plt.show()
```

If you would like to run a multi-node parallel high-throughput computation use the serial version instead by replacing the compute line as below:
```python
output = phase.serialcompute(3, configuration,meshsize )
```


Notes:
---------
This repository is arranged as follows : `/polyphase` contains the python codes required to compute phase diagram and visualization scripts. `expts` contains the scripts that were run on UB CCR, `notebooks` contains various test cases and example usage (some may be outdated and might not work)