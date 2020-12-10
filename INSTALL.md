## Installing polyphase package
We will go over the details of installing `polyphase` as a python package using `pip` and `conda`

We recommend installing using conda environments but advanced user can use python virtual environment set up described [here](https://docs.python.org/3/tutorial/venv.html)

To install using conda, first install python using [Anaconda](https://docs.anaconda.com/anaconda/install/)

After sucessfull installation follow the steps

1. In a terminal, create an environment using
```bash
conda create --name phasemodel python=3.7
```

2. Activate the conda environment you just created using:
```bash
conda activate phasemodel
```

3. Clone the github master repository of the `polyphase` module using:
```bash
git clone https://github.com/kiranvad/polyphase.git
```
Change the current directory to the newly cloned github repo.

4. Install the dependencies of polyphase using:
```bash
pip install -r requirements.txt
```

5. Install the polyphase as a package:
```python
pip install -e .
```

You should be able to repeat similar steps on python virtual environment if you chose to.
The commands above would also install several dependecies for example `ray`, `mpltern` thus it might take about a minute.
