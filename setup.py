from setuptools import setup,find_packages
import sys, os

setup(name="polyphase",
      description="Polymer thermodynamic phase modelling",
      version='1.0',
      author='Kiran Vaddi',
      author_email='kiranvad@buffalo.edu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['numpy',
      'scipy',
      'matplotlib', 'pandas', 'ray', 'plotly', 'mpltern', 'packaging'],
      extras_require = {},
      packages=find_packages(),
      long_description=open('readme.md').read(),
)