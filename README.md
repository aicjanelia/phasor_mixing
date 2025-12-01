# THIS README IS A WORK IN PROGRESS
Further instructions will be added shortly.

# Phasor Mixing Coefficient.
Implementation of a phasor-based method for analyzing signal mixing in multispectral images.

This code accompanies a manuscript currently under review. The manuscript_figures folder contains all relevant analysis code to produce the figures for the manuscript. The Python file pmc/pmc.py contains helper functions and the main classes--Phasor, PC, and PairMixture--which handle the phasor analysis of signal mixing.

The examples provided in the manuscript_figures folder provide illustrative instruction for using this method. However, a more general and detailed explanation will be provided here in the future.

# Notes
This code was built and run on a PC running Windows 11 and WSL (Ubuntu).  
The machine in question was equipped with the following relevant hardware:
* Intel(R) Xeon(R) w7-3465X 2.50 GHz
* 512 GB RAM  

No GPU is required and the code was not optimized for utilizing a GPU.  

# Installation
Clone this repository into a local directory:
```
git clone https://github.com/aicjanelia/phasor_mixing
```
Create a new Python environment with your preferred environment manager (such as miniforge)
```
conda create -n phasor_mixing python=3.10.12
conda activate phasor_mixing
```
Install the required dependencies using pip
```
python -m pip install -r phasor_requirements.txt
```

# Access to data for examples
The data relevant for the manuscript can be accessed at FigShare.  
The example Jupyter notebook will require the user to edit the path to point to the directory containing the downloaded data, as indicated.
