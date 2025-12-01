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

# Example
Running PMC analysis consists of calling only a handful of functions. Default parameters were used for all parameters throughout the associated manuscript.

The first step of PMC analysis is to determine the pure components (PC's). To do so, we first load the multispectral images and compute the phasor transform.
```
pc1_test = PC(file='/path/to/single_color_control_1.czi',name='Control_1',thresh=None,filt=True)
pc2_test = PC(file='/path/to/single_color_control_2.czi',name='Control_2',thresh=None,filt=True)
```
Then a two-step clustering workflow determines the centroids of the phasor clusters for each PC.
```
pc1_test.determine_peak()
pc2_test.determine_peak()
```
After PC phasor coordinates are determined, we load the two-color experiment image and compute its phasor transform. 
The single-color control objects--with their computed PC coordinates--are passed to the the mixture object as variables.
```
crop_dims = [y1,y2,x1,x2]

mix_mitomito = PairMixture(file='path/to/two_color_cell_1.czi',name='mito',
                  pcs=[pc1_test,pc2_test],thresh=None,filt=True,crop_dims=crop_dims)
```
Once the two-color multispectral image is converted to phasor space, the PC coordinates are used to define the mixing band.
PMC is then calculated by taking the mean and normalized variance of the distribution of points within this band.
```
mix_mitomito.analyze_overlap()
```
For comparison throughout the associated manuscript, Pearson's and Manders' Correlation Coefficients (PCC and M1, M2) are also calculated.
```
mix_mitomito.calc_pearson_mander(pcs=mix_mitomito.purecomps)
```

# Access to data for examples
The data relevant for the manuscript can be accessed at FigShare (doi in process).  
The example Jupyter notebook will require the user to edit the path to point to the directory containing the downloaded data, as indicated.
