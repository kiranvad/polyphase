# Phase Modelling Using Convex Envelope Method

Installation instructions can be found in [INSTALL.md](/INSTALL.md)

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
engine.plot()
```

If you would like to run a multi-node parallel high-throughput computation use the serial version instead by replacing the compute line as below:
```python
engine.compute(use_parallel=True, verbose=True, lift_label=True)
```

To run a simple example in jupyter use the following [example](/notebooks/example.ipynb)
