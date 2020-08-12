# Phase Modelling Convex Envelope method

To install, run the following commands (with git and python pip installed)
```bash
git clone https://github.com/kiranvad/polyphase.git
cd polyphase
```

Install `polyphase` as a Python package:
```python
pip install -e .
```
This will instal polyphase and other dependencies namely : numpy, scipy, matplotlib, pandas and ray (for parallel comutation), mpltern (for ternary phase diagram visualization), plotly(for intereactive energy landscape visualization)

A sample use case is as follows:

```python
import polyphase as phase
import ray
ray.init()

M = [5,5,1]
chi = [0.5, 0.5, 1]
configuration = {'M':M, 'chi':chi}
meshsize = 100
output = phase.compute(3, configuration,meshsize ) 
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

If you would like to run a multi-node parallel high-throughput computation use the serial version by replacing the compute line as below:
```python
output = phase.serialcompute(3, configuration,meshsize )
```


Notes:
---------
Currently support multi-node parallelization using ray.
