import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import os
import re
import pickle
import colorcet as cc
import tifffile
from aicsimageio import AICSImage

from scipy.io import loadmat
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.signal import peak_widths, find_peaks
from scipy.optimize import lsq_linear, nnls
from scipy.stats import gaussian_kde

from sklearn.cluster import HDBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA

from skimage.feature import peak_local_max
from skimage.morphology import disk, ball
from skimage.filters import threshold_otsu, threshold_mean, threshold_isodata, threshold_triangle, threshold_li, median
from skimage.exposure import rescale_intensity
from skimage.measure import pearson_corr_coeff, manders_coloc_coeff, manders_overlap_coeff

from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from functools import partial

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

import matplotlib as mpl
import colorcet
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def im2DLUT(Ac,Aint,cmap=None,cLim=None,intLim=None):
    
    if cmap is None:
        obj = 'magma'
        cmap = mpl.colormaps.get_cmap(obj)

    if cLim is None:
        cLim = np.array([np.amin(Ac), np.amax(Ac)])
        
    if intLim is None:
        intLim = np.percentile(Aint, [0.00,99.5])
        
    Nc = cmap.N
    divMap = rgb_to_hsv(np.array([cmap(i)[:-1] for i in range(Nc)]))
    divMap = np.reshape(divMap,[Nc,1,3])
    divMap = np.repeat(divMap,Nc,axis=1)

    obj = 'cet_CET_L1'
    intMap = mpl.colormaps.get_cmap(obj) 
    inT = np.mean(np.array([intMap(i)[:-1] for i in range(Nc)]),axis=1)
    divMap[...,-1] = np.repeat(inT[np.newaxis,:],Nc,axis=0)

    divMap = hsv_to_rgb(divMap)
    
    hueBins = np.linspace(cLim[0],cLim[-1],num=256)
    hueBins[0] = -np.inf
    hueBins[-1] = np.inf
    
    hue = np.digitize(Ac,hueBins)
    
    briBins = np.linspace(intLim[0],intLim[-1],num=256)
    briBins[0] = -np.inf
    briBins[-1] = np.inf
    
    bri = np.digitize(Aint,briBins)
    
    badVals = np.logical_or(np.isnan(hue),np.isnan(bri))
    hue[badVals] = 1
    bri[badVals] = 1

    LUTim = np.zeros_like(Ac)
    LUTim = np.repeat(LUTim[...,np.newaxis],3,axis=-1)
    LUTim[...,:] = divMap[np.newaxis,hue[...],bri[...],:]

    return divMap, LUTim


def im2DLUT_quad(Ac,Aint,cmap=None,cLim=None,intLim=None):
    
    if cmap is None:
        # obj = 'cet_linear_kry_0_97_c73'
        obj = 'magma'
        cmap = mpl.colormaps.get_cmap(obj)

    if cLim is None:
        cLim = np.array([np.amin(Ac), np.amax(Ac)])
        
    if intLim is None:
        intLim = np.percentile(Aint, [0.00,99.5])
        
    Nc = cmap.N
    divMap = rgb_to_hsv(np.array([cmap(i)[:-1] for i in range(Nc)]))
    divMap = np.reshape(divMap,[Nc,1,3])
    divMap = np.repeat(divMap,Nc,axis=1)
    obj = 'cet_CET_L1'
    intMap = mpl.colormaps.get_cmap(obj) 
    inT = np.mean(np.array([intMap(i)[:-1] for i in range(Nc)]),axis=1)
    divMap[:,:,-1] = np.outer(inT,inT)
    divMap = hsv_to_rgb(divMap)
    
    hueBins = np.linspace(cLim[0],cLim[-1],num=256)
    hueBins[0] = -np.inf
    hueBins[-1] = np.inf
    
    hue = np.digitize(Ac,hueBins)
    
    briBins = np.linspace(intLim[0],intLim[-1],num=256)
    briBins[0] = -np.inf
    briBins[-1] = np.inf
    
    bri = np.digitize(Aint,briBins)
    
    badVals = np.logical_or(np.isnan(hue),np.isnan(bri))
    hue[badVals] = 1
    bri[badVals] = 1
    
    LUTim = np.zeros_like(Ac)
    LUTim = np.repeat(LUTim[...,np.newaxis],3,axis=-1)
    LUTim[...,:] = divMap[np.newaxis,hue[...],bri[...],:]
    
    return divMap, LUTim

plt.style.use('default')
colors = ['#EC8609','#0365B5','#5EB5FD','#671343','#DF53A2']


class InductiveClusterer(BaseEstimator):
    '''
    # Authors: Chirag Nagpal
    #          Christos Aridas

    '''

    def __init__(self, clusterer, classifier):
        self.clusterer = clusterer
        self.classifier = classifier
    
    @staticmethod
    def _classifier_has(attr):
        """Check if we can delegate a method to the underlying classifier.

        First, we check the first fitted classifier if available, otherwise we
        check the unfitted classifier.
        """
        return lambda estimator: (
            hasattr(estimator.classifier_, attr)
            if hasattr(estimator, "classifier_")
            else hasattr(estimator.classifier, attr)
        )

    def fit(self, X, y=None):
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y = self.clusterer_.fit_predict(X)
        self.classifier_.fit(X, y)
        return self

    @available_if(_classifier_has("predict"))
    def predict(self, X):
        check_is_fitted(self)
        return self.classifier_.predict(X)
    
    @available_if(_classifier_has("predict_proba"))
    def predict_proba(self, X):
        check_is_fitted(self)
        return self.classifier_.predict_proba(X)

    @available_if(_classifier_has("decision_function"))
    def decision_function(self, X):
        check_is_fitted(self)
        return self.classifier_.decision_function(X)

class Phasor:
    ''' Phasor analysis for hyperspectral imaging
    
    Parameters:
        name : str
            name of the sample or dataset

                Serves no inherent function, merely for annotation
                
        file : str
            file name for hyperspectral intensity data 
            
                Currently accepts .czi , .tif and .csv formats
            
                    Files in csv format require the structure [lambdas, int for n pixels]
                        where lambdas and int are column arrays in a matrix of size
                        n_lambdas x n_pixels (no current support for 3D)
                        
        precomp_flag : bool
            flag to indicate if the file contains pre-computed phasor coordinates
            
                Used when file type = .csv
        
        harm : int
            phasor harmonic to compute (default = 1)

                No harmonic greater than 1 is used within this manuscript
        
        thresh : str
            name of intensity threshold calculation method
            
                Options : otsu, mean, li, iso, triangle
                
        filt : bool
            option to apply median filter to phasor coordinations of kernel size 3 (default True)
            
        mask : MxN binary array
            optional pre-computed binary mask for the input image
            
                Must be same shape (MxN) of one slice of the input image 

        min_lambda : int
            lower-bound for wavelengths to include in the phasor transform
            
                To compare across samples for which the range of acquired wavelength differs, adjust this
                parameter to normalize the phasor plot
                
        ver : int
            choice of normalization for the phasor transform (default 0)

                This should not be changed from default
                
        data_arr : MxN array
            MxN array representing a lambda stack
            
                Largely used to testing purposes, in case one does not have a separate file in a supported format 
                that corresponds to the lambda stack
                
        lambda_arr : 1xN array
            array of size N containing the wavelengths corresponding to the data in data_arr
            
                Used only in conjunction with data_arr
                
        crop_dims : DxD array
            array containing the coordinates for cropping the input image (default: no cropping)
                
                D is the dimension of the image file (almost always D = 2)
        
    Methods
        plot_phasor : plot the 2D density scatter plot of phasor coordinates

        parameters :
            ax1 : pyplot axis object 
            
            p_max : upper density limit that corresponds to maximum brightness [0.0,1.0]
            
                If phasor clusters are not visibile, decrease p_max
    '''
    cmap = lsm.from_list(name = 'test', colors=['0','#DF53A2'])
    @staticmethod
    def integrate_(pixel,lam,harm=1,ver=0):
        ''' 
        Helper function to perform phasor transform
        Parameters : 
        
        pixel : 1xN array
            array containing intensity counts over a range of wavelengths
            
        lam : 1xN array
            array of wavelengths corresponding to pixel
            
        harm : int
            harmonic for the phasor transform (default = 1)
            
        ver : int
            choice of normalization for the phasor transform (default = 0)
        '''
        if pixel.ndim == 2 :   
            pixel_n = pixel - np.amin(pixel,axis=-1)[...,None]
        else:
            pixel_n = pixel - np.amin(pixel,axis=-1)[...,None]

        if ver==1:
            S_num = np.trapz(pixel_n*np.sin(harm*(lam)*(2*np.pi/(np.amax(lam)-np.amin(lam)))),lam,axis=-1)
            G_num = np.trapz(pixel_n*np.cos(harm*(lam)*(2*np.pi/(np.amax(lam)-np.amin(lam)))),lam,axis=-1)
            denom = np.trapz(pixel_n,lam)
        else:
            S_num = np.trapz(pixel_n*np.sin(harm*(lam-np.amin(lam))*(2*np.pi/(np.amax(lam)-np.amin(lam)))),lam,axis=-1)
            G_num = np.trapz(pixel_n*np.cos(harm*(lam-np.amin(lam))*(2*np.pi/(np.amax(lam)-np.amin(lam)))),lam,axis=-1)
            denom = np.trapz(pixel_n,lam)
        result = np.nan_to_num(np.array([np.divide(G_num, denom, out=np.zeros_like(G_num), where=denom!=0),
                                np.divide(S_num, denom, out=np.zeros_like(S_num), where=denom!=0)]),posinf=0.0,neginf=0.0)
        return result
    
    def __init__(self,  name, file, precomp_flag=False, harm=1,thresh=None,filt=False,mask=None,min_lambda=None,
                 ver=0,data_arr=None,lambda_arr=None,crop_dims=None):

        self.name = name
        self.file = file
        if thresh is not None:
            self.thresh_flag = True
            self.thresh_method = thresh
        else:
            self.thresh_flag = False
            self.thresh_method = None
        
            
        ### Load multispectral data from some existing array
        if file is None and data_arr is not None:
            self.data_cyx = np.moveaxis(data_arr,-1,0)
            self.intensity_orig = np.moveaxis(self.data_cyx,0,-1)
            self.lambdas = lambda_arr
            
            if mask is not None:
                binary = mask > 0.0
                data_cyx[:,~binary] = np.nan
            else: 
                if self.thresh_flag:
                    temp_im = np.sum(np.moveaxis(self.data_cyx,0,-1),axis=-1)
                    temp_im = median(temp_im, disk(5))
                    if self.thresh_method == 'otsu':
                        self.thresh_val = threshold_otsu(temp_im)
                    elif self.thresh_method == 'li':
                        self.thresh_val = threshold_li(temp_im)
                    elif self.thresh_method == 'mean':
                        self.thresh_val = threshold_mean(temp_im)
                    elif self.thresh_method == 'triangle':
                        self.thresh_val = threshold_triangle(temp_im)
                    elif self.thresh_method == 'iso':
                        self.thresh_val = threshold_isodata(temp_im)
                    else:
                        print('Threshold method not regognized. Using Otsu')
                        self.thresh_val = threshold_otsu(temp_im)
                    
                    self.binary = temp_im < self.thresh_val
                    self.data_cyx[:,self.binary] = np.nan
                else:
                    self.thresh_method = None

            self.intensity = np.moveaxis(self.data_cyx,0,-1)

            size = np.shape(self.data_cyx)
            lambdas_temp = np.tile(self.lambdas,(size[1],1))
            lambdas_new = np.repeat(lambdas_temp.T[:,:,np.newaxis],size[2],2)
            self.lambdas_new = 1.0*np.moveaxis(lambdas_new,0,-1)

            self.phasor_coords = np.nan_to_num(np.moveaxis(Phasor.integrate_(self.intensity,self.lambdas_new,harm=harm,ver=ver),0,-1))
            
            if filt:
                print('Filtering Phasor Coordinates...')
                self.phasor_coords[:,:,0] = median(self.phasor_coords[:,:,0], disk(3))
                self.phasor_coords[:,:,1] = median(self.phasor_coords[:,:,1], disk(3))
            
        else:
            file_split = os.path.splitext(file)
            if file_split[1] in ['.czi','.tif']:
                
                try:
                    self.img = AICSImage(file)
                except:
                    print('Could not load file.')
                    print('Make sure path points to BioImageFormat (.czi) file')
                
                else:

                    if file_split[1] == '.czi':
                        if self.img.dims['Z'][0] > 1:
                            if crop_dims is not None:
                                self.data_cyx = np.array(self.img.get_image_data("CZYX", S=0, T=0),dtype=np.float32)[:,crop_dims[0]:crop_dims[1],crop_dims[2]:crop_dims[3],crop_dims[4]:crop_dims[5]]
                            else:
                                self.data_cyx = np.array(self.img.get_image_data("CZYX", S=0, T=0),dtype=np.float32)
                        else:
                            if crop_dims is not None:
                                self.data_cyx = np.array(self.img.get_image_data("CYX", Z=0, S=0, T=0),dtype=np.float32)[:,crop_dims[0]:crop_dims[1],crop_dims[2]:crop_dims[3]]
                            else:
                                self.data_cyx = np.array(self.img.get_image_data("CYX", Z=0, S=0, T=0),dtype=np.float32)
                                
                        pattern = re.compile(r'\d{3}')
                        self.lambdas = np.array([float(re.match(pattern,ch_name).group()) for ch_name in self.img.channel_names])
                    elif file_split[1] == '.tif':
                        with tifffile.TiffFile(file) as tif:
                            imagej_metadata = tif.imagej_metadata
                        self.lambdas = np.array([float(st) for st in imagej_metadata['Labels']])
                        
                        if self.img.dims['Z'][0] > 1:
                            if crop_dims is not None:
                                self.data_cyx = np.array(self.img.get_image_data("ZCYX", S=0, T=0),dtype=np.float32)[:,crop_dims[0]:crop_dims[1],crop_dims[2]:crop_dims[3],crop_dims[4]:crop_dims[5]]
                                self.data_cyx = np.moveaxis(self.data_cyx,0,1)
                            else:
                                self.data_cyx = np.array(self.img.get_image_data("ZCYX", S=0, T=0),dtype=np.float32)
                                self.data_cyx = np.moveaxis(self.data_cyx,0,1)
                        else:
                            if crop_dims is not None:
                                self.data_cyx = np.array(self.img.get_image_data("CYX", Z=0, S=0, T=0),dtype=np.float32)[:,crop_dims[0]:crop_dims[1],crop_dims[2]:crop_dims[3]]
                                
                            else:
                                self.data_cyx = np.array(self.img.get_image_data("CYX", Z=0, S=0, T=0),dtype=np.float32)
                                
                    if min_lambda is not None:

                        self.data_cyx = self.data_cyx[self.lambdas>=min_lambda,...]
                        self.lambdas = self.lambdas[self.lambdas>=min_lambda]
                    self.intensity_orig = 1.0*np.moveaxis(self.data_cyx,0,-1)
                    
                    if mask is not None:
                        binary = mask > 0.0
                        data_cyx[:,~binary] = np.nan
                        
                    else: 
                        if self.thresh_flag:
                            temp_im = np.sum(np.moveaxis(self.data_cyx,0,-1),axis=-1)
                            if self.img.dims['Z'][0] > 1:
                                kernel = ball(5)
                            else:
                                kernel = disk(5)

                            temp_im = median(temp_im, kernel)
                            if self.thresh_method == 'otsu':
                                self.thresh_val = threshold_otsu(temp_im)
                            elif self.thresh_method == 'li':
                                self.thresh_val = threshold_li(temp_im)
                            elif self.thresh_method == 'mean':
                                self.thresh_val = threshold_mean(temp_im)
                            elif self.thresh_method == 'triangle':
                                self.thresh_val = threshold_triangle(temp_im)
                            elif self.thresh_method == 'iso':
                                self.thresh_val = threshold_isodata(temp_im)
                            else:
                                print('Threshold method not regognized. Using Otsu')
                                self.thresh_val = threshold_otsu(temp_im)
                            
                            self.binary = temp_im < self.thresh_val
                            self.data_cyx[:,self.binary] = np.nan
                        else:
                            self.thresh_method = None
                        
                        
                    self.intensity = 1.0*np.moveaxis(self.data_cyx,0,-1)
                    
                    size = np.shape(self.data_cyx)
                    lambdas_temp = np.tile(self.lambdas,(size[1],1))
                    if self.img.dims['Z'][0] > 1:
                        lambdas_new = lambdas_temp.T[...,np.newaxis,np.newaxis].repeat(size[-1],2).repeat(size[-2],3)
                    else:
                        lambdas_new = np.repeat(lambdas_temp.T[:,:,np.newaxis],size[2],2)
                    self.lambdas_new = 1.0*np.moveaxis(lambdas_new,0,-1)
                    
                    self.phasor_coords = np.nan_to_num(np.moveaxis(Phasor.integrate_(self.intensity,self.lambdas_new,harm=harm,ver=ver),0,-1))
                    if filt:
                        print('Filtering Phasor Coordinates...')
                        kernel = disk(3)
                        if self.img.dims['Z'][0] > 1:

                            for j in range(self.phasor_coords.shape[0]):
                                self.phasor_coords[j,...,0] = median(self.phasor_coords[j,...,0], footprint=kernel)
                                self.phasor_coords[j,...,1] = median(self.phasor_coords[j,...,1], footprint=kernel)
                        else:

                            self.phasor_coords[...,0] = median(self.phasor_coords[...,0], footprint=kernel)
                            self.phasor_coords[...,1] = median(self.phasor_coords[...,1], footprint=kernel)

            
            elif file_split[1]=='.csv':
                if not precomp_flag:
                    
                    try:
                        data = np.loadtxt(file,delimiter=',')
                    except:
                        print('Could not load file.')
                        print('Make sure path points to csv (.csv) file')
                    
                    else:
                        self.lambdas = data[:,0]
                        self.img = data[:,1:]
                        shape = self.img.shape
                        n_lam = shape[0]
                        n_pix = int(np.sqrt(shape[1]))
                        indx = np.unravel_index(range(shape[1]),(n_pix,n_pix))
                        self.data_cyx = np.zeros((n_lam,n_pix,n_pix))
                        for j in range(shape[1]):
                            self.data_cyx[:,indx[0][j],indx[1][j]] = self.img[:,j].T
                        self.intensity_orig =  1.0*np.moveaxis(self.data_cyx,0,-1)
                        if self.thresh_flag:
                            self.thresh_method = thresh
                            temp_im = np.sum(np.moveaxis(self.data_cyx,0,-1),axis=-1)
                            temp_im = median(temp_im, disk(5))
                            if self.thresh_method == 'otsu':
                                self.thresh_val = threshold_otsu(temp_im)
                            elif self.thresh_method == 'li':
                                self.thresh_val = threshold_li(temp_im)
                            elif self.thresh_method == 'mean':
                                self.thresh_val = threshold_mean(temp_im)
                            elif self.thresh_method == 'triangle':
                                self.thresh_val = threshold_triangle(temp_im)
                            elif self.thresh_method == 'iso':
                                self.thresh_val = threshold_isodata(temp_im)
                            else:
                                print('Threshold method not regognized. Using Otsu')
                                self.thresh_val = threshold_otsu(temp_im)
                            
                            self.binary = temp_im < self.thresh_val
                            self.data_cyx[:,self.binary] = np.nan
                            
                        self.intensity =  1.0*np.moveaxis(self.data_cyx,0,-1)

                        size = np.shape(self.data_cyx)
                        lambdas_temp = np.tile(self.lambdas,(size[1],1))
                        lambdas_new = np.repeat(lambdas_temp.T[:,:,np.newaxis],size[2],2)
                        self.lambdas_new = 1.0*np.moveaxis(lambdas_new,0,-1)
                        
                        self.phasor_coords = np.nan_to_num(np.moveaxis(Phasor.integrate_(self.intensity,self.lambdas_new,harm=harm,ver=ver),0,-1))
                        if filt:
                            print('Filtering Phasor Coordinates...')
                            self.phasor_coords[:,:,0] = median(self.phasor_coords[:,:,0], disk(3))
                            self.phasor_coords[:,:,1] = median(self.phasor_coords[:,:,1], disk(3))
                else:
                    try:
                        data = np.loadtxt(file,delimiter=',')
                    except:
                        print('Could not load file.')
                        print('Make sure path points to csv (.csv) file')
                        
                    else:
                        n_lam = 32
                        self.lambdas = range(n_lam) ### placeholder
                        self.img = data
                        shape = self.img.shape
                        self.thresh_method = thresh
                        
                        n_pix = int(np.sqrt(shape[0]))
                        indx = np.unravel_index(range(shape[0]),(n_pix,n_pix))
                        self.data_cyx = np.zeros((n_lam,n_pix,n_pix))
                        self.intensity = np.moveaxis(self.data_cyx,0,-1)
                        if filt:
                            self.intensity = median(self.intensity, ball(2))
                        self.phasor_coords = np.zeros((n_pix,n_pix,2))
                        for j in range(shape[0]):
                            self.phasor_coords[indx[0][j],indx[1][j],0] = data[j,0]
                            self.phasor_coords[indx[0][j],indx[1][j],1] = data[j,1]
            
                
            else:
                print('Error: File is not in a readable format...')
             
    
    def plot_phasor(self,ax1=None,cmap=cmap,p_max=1.0,plt_save=False):
        if ax1 is None:
            fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(5,5))
        if self.thresh_flag:
            pc_x = self.phasor_coords[~self.binary,0]
            pc_y = self.phasor_coords[~self.binary,1]
        else:
            # pc_x = self.phasor_coords[:,:,0]
            # pc_y = self.phasor_coords[:,:,1]   
            pc_x = self.phasor_coords[...,0]
            pc_y = self.phasor_coords[...,1]   
        sns.histplot(ax=ax1,x=pc_x.flatten(),y=pc_y.flatten(),cmap=cmap,zorder=0,stat='density',bins=100,pmax=p_max)
        c1=plt.Circle((0,0),radius=1,color='k',fill=True,zorder=-100)
        ax1.add_patch(c1)
        ax1.set_xlim([-1.05,1.05]); ax1.set_ylim([-1.05,1.05]); 
        ax1.set_xticks([]); ax1.set_yticks([]);
        ax1.set_aspect('equal'); 
        ax1.set_xlabel(r'$G$'); ax1.set_ylabel(r'$S$');
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        if plt_save:
            plt.savefig(name+'_phasor_plot.svg',dpi=600)
        return ax1
            
class PC(Phasor):
    '''
    Pure component implementation of Phasor Analysis
    
    Parameters:
        
        Inherited parameters from Phasor class : name, file, precomp_flag, harm,data_arr, ver, lambda_arr, min_lambda, thresh, filt, mask, crop_dims
        
    Methods:
        cluster_pc : determine background (near origin), pc (phasor point), and noise (other) distributions
        
            Method :
                This is a two-step process. First, the entire phasor cloud is roughly clustered into n_comp
                elliptical clusters using Gaussian Mixture Models. 
                If an intensity threshold was applied at initialization, this first step is skipped.
                The rough foreground cluster is taken to be the one that sits furthest from the origin.
                This rought foreground cluster is then refined using HDBSCAN.
                At each step, the clustering is performed on set_size * number of pixels. This is to save
                computation time as clustering large datasets can be prohibitively slow.
                An inductive classifier is then used at each step to assign each pixel to the most likely cluster.

            parameters (defaults should work in most cases) :
                n_comp : number of elliptical clusters expected for GMM clustering
                
                    If the sample exhibits significant background staining of distinct spectra,
                    setting n_comp > 2 might be necessary
                
                set_size : fraction of pixels to use (1% is default, 10% runs very slowly)
                
                min_pts : minimum number of points to define a cluster

                    See HDBSCAN docs for more information
        
        determine_peak : find peaks of background and pc distributions using KDE 

            Method : 
                Kernel Density Estimation calculates the estimated density of points throughout the cluster.
                The center of the cluster is determined by one of two methods determined by the 'pk_flag' parameter.
                If pk_flag == 'test':
                    The rectangular boundary of the cluster is filled with a uniform mesh of points. The KDE estimator
                    calculates the log density at each of these points.
                    The center of the cluster is defined to be the peak of this density profile.
                else :
                    The KDE estimate is directly used as the density profile.
                    The center of the cluster is defined to be the density-weighted centroid.
                    
                The two methods are extremely similar. The 'test' method is more robust as it is independent of the
                distribution of the phasor points themselves.

            Inherited parameters from cluster_pc : n_comp, set_size, min_pts
            
            parameters (defaults should work in most cases) :

                pk_flag : str (default : 'test')
                    Only one option. If not 'test', then the less-robust method is used.
                
                bandwidth : KDE bandwidth (default 0..01)
                    A smaller number gives more accurate peak determination. The default is almost always sufficient.
                
        plot_clusters, plot_hist_with_peaks : plotting helpers    
        
    '''
    
    def __init__(self, name,file=None,precomp_flag=False,harm=1,data_arr = None, 
                 ver=0,lambda_arr = None, min_lambda = None, thresh=None,filt=False,mask=None,crop_dims=[0,-1,0,-1]):

        super().__init__(file=file,name=name,precomp_flag=precomp_flag,ver=ver,harm=harm,data_arr = data_arr, 
                         lambda_arr = lambda_arr, min_lambda = min_lambda, thresh=thresh,filt=filt,crop_dims=crop_dims)
            
    
    def cluster_pc(self,n_comp=2,set_size=0.01,min_pts=0.005,plt_save=False,sim_flag=False,no_bg = False):
        if self.thresh_flag:
            self.no_bg = True
        else:
            self.no_bg = no_bg
        if not hasattr(self,'X'):

            # data_x = self.phasor_coords[:,:,0].flatten(); data_y = self.phasor_coords[:,:,1].flatten()
            data_x = self.phasor_coords[...,0].flatten(); data_y = self.phasor_coords[...,1].flatten()
            self.X = np.stack((data_x, data_y),axis=1)
        y = np.zeros(np.shape(self.X)[0])
        self.X_train, self.X_test, y_train, y_test = train_test_split(self.X, y, test_size=set_size, random_state=42)
        
        if not self.no_bg:

            self.GMM = GaussianMixture(n_components=n_comp)

            classifier = RandomForestClassifier(random_state=42,n_jobs = int(0.875 * cpu_count()))
            inductive_learner = InductiveClusterer(self.GMM, classifier).fit(self.X_test)

            self.clusters_GMM = inductive_learner.predict(self.X)
            
            self.Xmeans = np.array([[np.mean(self.X[self.clusters_GMM==ci,0]),
                                    np.mean(self.X[self.clusters_GMM==ci,1])] for ci in np.unique(self.clusters_GMM)])
            self.cmean = np.unique(self.clusters_GMM)[np.argmax(cdist(self.Xmeans,[[0.0,0.0]]))]
            
            self.data_c = self.X[self.clusters_GMM==self.cmean,:]
        else:
            self.data_c = self.X

        y = np.zeros(np.shape(self.data_c)[0])
        X_train, self.X_test_c, y_train, y_test = train_test_split(self.data_c, y, test_size=set_size, random_state=42)

        self.HD = HDBSCAN(min_samples=np.amax([int(np.shape(self.X_test_c)[0]*min_pts),5]),allow_single_cluster=True).fit(self.X_test_c)

        classifier = RandomForestClassifier(random_state=42,n_jobs = int(0.875 * cpu_count()))
        self.inductive_learner = InductiveClusterer(self.HD, classifier).fit(self.X_test_c)

        self.clusters = self.inductive_learner.predict(self.data_c)
        
        c_centers = np.zeros(len(np.unique(self.clusters[self.clusters>=0])))
        for c, j in zip(np.unique(self.clusters[self.clusters>=0]),range(len(np.unique(self.clusters[self.clusters>=0])))):
            center_x = np.mean(self.data_c[self.clusters==c,0])
            center_y = np.mean(self.data_c[self.clusters==c,1])
            origin_dist = np.sqrt(center_x**2 + center_y**2)
            c_centers[j] = origin_dist
            
        cluster_ids = np.unique(self.clusters[self.clusters>=0])
        self.cluster_id = cluster_ids[np.argmax(c_centers)]
        print('Finished clustering data for ' + self.name + '...')
        
        return self

    def plot_clusters(self,ax1=None,cmap=Phasor.cmap,plt_save=False):
        if ax1 is None:
            fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(5,5))
        colors = ["g.", "r.", "b.", "y.", "c."]
        if not self.no_bg:
            for klass, color in zip(np.unique(self.clusters[self.clusters>=0]), colors):
                Xk = self.data_c[self.clusters == klass]
                sns.histplot(ax=ax1,x=Xk[:, 0],y=Xk[:, 1],cmap=Phasor.cmap,zorder=-10)
        else:
            for klass, color in zip(np.unique(self.clusters[self.clusters>0]), colors):
                Xk = self.data_c[self.clusters == klass]
                sns.histplot(ax=ax1,x=Xk[:, 0],y=Xk[:, 1],cmap=cmap,zorder=-10)
                
        c1=plt.Circle((0,0),radius=1,color='k',fill=True,zorder=-100)
        ax1.add_patch(c1)
        ax1.set_xlim([-1,1]); ax1.set_ylim([-1,1]); ax1.set_aspect('equal'); ax1.set_title(self.name);
        ax1.set_xlabel(r'$G$'); ax1.set_ylabel(r'$S$');
        if plt_save:
            plt.savefig(self.name+'_pc_cluster_plot.svg',dpi=600)
                
        return ax1
    
    @staticmethod
    def parrallel_score_samples(kde, samples, thread_count=int(0.875 * cpu_count())):
        with Pool(thread_count) as p:
            return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))
        
    @staticmethod
    def make_ellipses(gmm):
        for n, color in enumerate(colors):
            if gmm.covariance_type == "full":
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == "tied":
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == "diag":
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == "spherical":
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
            v, w = np.linalg.eigh(covariances)

            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            
            return v
        
    def determine_peak(self,n_comp=2,bandwidth=0.01,set_size=0.01,min_pts=0.005,plot=True,plt_save=False, sim_flag=False,pk_flag='test',no_bg=False):
        
        if not hasattr(self,'X_test'):
            self.cluster_pc(n_comp=n_comp,set_size=set_size,min_pts=min_pts,sim_flag=sim_flag,no_bg=no_bg)
        
        y = np.zeros(np.shape(self.data_c[self.clusters==self.cluster_id])[0])
        X_train, X_test_c, y_train, y_test = train_test_split(self.data_c[self.clusters==self.cluster_id], y, test_size=0.01, random_state=42)
        self.kde = KernelDensity(bandwidth=bandwidth,atol=0.0005,rtol=0.0001).fit(X_test_c)

        self.pc_cluster = self.data_c[self.clusters==self.cluster_id]
        self.density = PC.parrallel_score_samples(self.kde,self.pc_cluster)
        pca = PCA(n_components=2)
        model = pca.fit(self.pc_cluster)
        temp_cov = model.get_covariance()
        v, w = np.linalg.eigh(temp_cov)

        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        self.cov = np.amax(v)

        center_x = np.sum(self.pc_cluster[:,0]*np.exp(self.density))/np.sum(np.exp(self.density))
        center_y = np.sum(self.pc_cluster[:,1]*np.exp(self.density))/np.sum(np.exp(self.density))
        
        ### Slight different implementations of determining the center of the PC cluster.
        ### This choice results in slightly different centers which can be useful depending
        ### on the shape of the PC cluster.
        if pk_flag == 'test':
            npts = 50
            npts1 = 200
            x1 = np.sort(np.unique(np.concatenate((np.linspace(1.2*np.amin(self.X[:,0]),1.2*np.amax(self.X[:,0]),npts1),np.linspace(-1.0,1.0,npts)))))
            x2 = np.sort(np.unique(np.concatenate((np.linspace(1.2*np.amin(self.X[:,1]),1.2*np.amax(self.X[:,1]),npts1),np.linspace(-1.0,1.0,npts)))))

            xv,yv=np.meshgrid(x1,x2,indexing='xy')
            log_density = self.kde.score_samples(np.vstack((xv.flatten(),yv.flatten())).T)
            density = np.exp(log_density)

            temp = np.vstack((xv.flatten(),yv.flatten())).T
            
            pks = peak_local_max(np.reshape(density,np.shape(xv)),num_peaks=1)
                    
            self.pks_l = np.squeeze(np.dstack((x1[pks[:,1]],x2[pks[:,0]])))
        else:
            self.pks_l = np.array([center_x, center_y])
            
        dists = cdist(self.pks_l[None,:],self.pc_cluster)
        self.rad = np.mean(dists)
        
        print('Determined peak at ' + str(self.pks_l) + 'for ' + self.name)
        return self
        
        
    def plot_hist_with_peaks(self,ax1=None,cmap=Phasor.cmap,p_max=1.0,plt_save=False):
        if ax1 is None:
            fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(5,5))
        if self.thresh_flag:
            pc_x = self.phasor_coords[~self.binary,0]
            pc_y = self.phasor_coords[~self.binary,1]
        else:
            # pc_x = self.phasor_coords[:,:,0]
            # pc_y = self.phasor_coords[:,:,1]  
            pc_x = self.phasor_coords[...,0]
            pc_y = self.phasor_coords[...,1]  
        sns.histplot(ax=ax1,x=pc_x.flatten(),y=pc_y.flatten(),cmap=cmap,zorder=-10,stat='density',bins=100,pmax=p_max)
        ax1.scatter(self.pks_l[0], self.pks_l[1],c='w',marker='x')
        c1=plt.Circle((0,0),radius=1,color='k',fill=True,zorder=-100)
        ax1.add_patch(c1)
        ax1.set_xlim([-1.05,1.05]); ax1.set_ylim([-1.05,1.05]); 
        ax1.set_xticks([]); ax1.set_yticks([]);
        ax1.set_aspect('equal'); 
        ax1.set_xlabel(r'$G$'); ax1.set_ylabel(r'$S$');
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        if plt_save:
            plt.savefig(self.name+'_phasorcould_with_peaks.svg',dpi=600)
        return ax1

class PairMixture(Phasor):
    '''
    Mixed sample implementation of Phasor Analysis
    
    Parameters:
        
        Inherited parameters from Phasor class : name, file, precomp_flag, harm,data_arr, ver, lambda_arr, min_lambda, thresh, filt, mask, crop_dims
        
        
    Methods:
        calc_band : define the mixing band based on the maximal size of the (2 or 3) PC clusters
        
            Method :
                Determine the maximal size of the PC clusters. Multiply this size by a constant factor (n_sigma)
                Define vertices of a 4-sided polygon at the center of each PC +/- the scaled cluster size.
                All phasor points inside this polygon are considered "foreground" pixels

            parameters (default should work in most cases) :
                n_sigma : multiplicative factor for the relative size of the mixing band
                        Not recommended to change as this eliminates the data-driven nature of the method.
                        However, if well-reasoned, this can be increased to include more phasor points
        
        calc_fraction : calculate the fractional contribution of each pc at each pixel in a multispectral image
                        by solving a linear system of equations

            Inherited parameter from calc_band : n_sigma
            
            parameters : none
            
        rescale_af : phasor-based linear unmixing to remove background signal from a multispectral image
                
                Method : 
                    Solve the three-species linear system to calculate the fraction of background in each pixel.
                    Multiply this fraction by the mean intensity of the background-only image.
                    Take the cross product of this quantity with the known background spectrum.
                    The resulting image is the background component at each pixel.
                    Subtracting this element-wise from the original multispectral image results in a background-free image.
                    The method also recomputes the phasor coordinates from this background-subtracted image.
                    
                parameters : none
                    
        color_mixing_image : calculates the color mixing image (CMI)
        
                Method : 
                    Multiplies f_mix by the normalized summed intensity at each pixel
                
                parameters : 

                    imtype : str (default '2D')
                        Whether to use a 1D or 2D LUT to display the CMI
                            The 1D strategy maps the product of f_mix and normalized intesntiy to a 1D colormap
                            The 2D strategy maps the vector [f_mix, intensity] to a 2D LUT where the color represents
                            the mixing and the brightness represents the intensity
                            
                    cmap : str
                        Choice of colormap
                            Any map accessible from either maplotlib or colorcet can be specified by its name
                            
                    quad : bool
                        Whether to map low mixing to low color AND brightness in the 2D LUT
                            Necessary to correctly display colormaps which start at dark colors (black)
                            
                    lims : 1x2 int array
                        Defines the min and max values to display in the CMI
        
        analyze_overlap : overall function to measure PMC

            Method :
                Calls the functions calc_fraction and calc_coloc
        
            Inherited parameter from calc_band : n_sigma
            
            Inhereited parameter from color_mixing_mage : lims
            
            Inherited parameter from plot_phasor : p_max
            
            parameters : none
        
        calc_coloc : calculate PMC

            Method :
                Calculates PMC_1 by computing the expectation value of the estimated Mixing Distribution.
                Calculates PMC_2 by scaling the size of the phasor cluster within the mixing band such that it takes
                the value 1 when its size is equal to or smaller than a single PC, and 0 when it is equal to or larger
                that the theoretical size of the cluster formed by concatenating two the PC clusters.
                
            parameters : none
        
        calc_pearson_mander : calculate PCC and M1, M2 from estimated filter images of the multispectral image
        
            Method :
                For each spectrum corresponding to each PC, determine the peak wavelength and full-width, half-max.
                Use the peak +/- FWHM/2 to select a range of wavelengths in the multispectral image.
                Sum this image over lambda. The resulting two intensity images are the estimated filter images for each PC.
                Pearson's and Manders' coefficients are calculated from these images using the scikit-image implementation
                of each method.
                
            parameters :
                
                lims : 1x2 int array
                    
                    Contrast-adjusts the estimated filter images (default : min = 2%, max = 98%)
                    
                thresh_override : bool
                    
                    Flag to override the absence of thresholding for the multispectral image (default : False)
                    
                thresh_method : str
                    
                    If thresh_override == True, choose the intensity threshold method
                        Options : otsu, li, triangle, mean, iso    
        
                
        plot_clusters, plot_hist_with_peaks, plot_band, plot_fraction : plotting helpers    
    '''
    
    def __init__(self, name, pcs, file=None, precomp_flag = False, 
                 pk_flag=None,harm=1,thresh=None,mask=None,ver=0,data_arr = None, min_lambda = None,
                 lambda_arr = None,filt=False,crop_dims=[0,-1,0,-1]):

        super().__init__(file=file,name=name,precomp_flag=precomp_flag,harm=harm,thresh=thresh,
                            ver=ver,data_arr = data_arr, lambda_arr = lambda_arr,filt=filt,crop_dims=crop_dims,min_lambda=min_lambda)
        

                
        for pc in pcs:
            if not hasattr(pc,'pks_l'):
                pc.determine_peak()
            else:
                print(pc.name + ' is already clustered... Peak at ' + str(pc.pks_l))

            
        self.pc_pks = np.stack([pc.pks_l for pc in pcs])
        self.pc_clusters = [pc.pc_cluster for pc in pcs]

        self.purecomps = pcs
        
    
    def plot_clusters(self,ax1=None,plt_save=False):
        if ax1 is None:
            fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(5,5))
        ax1.scatter(self.X_test[:,0],self.X_test[:,1],c=self.clusters)
        c1=plt.Circle((0,0),radius=1,color='k',fill=True,zorder=-100)
        ax1.add_patch(c1)
        ax1.set_xlim([-1,1]); ax1.set_ylim([-1,1]); ax1.set_aspect('equal'); ax1.set_title(self.name);
        ax1.set_xlabel(r'$G$'); ax1.set_ylabel(r'$S$');
        if plt_save:
            plt.savefig(name+'_pc_cluster_plot.svg',dpi=600)
                
        return ax1
    
    def plot_hist_with_peaks(self,ax1=None,cmap=Phasor.cmap,plot_save=False,p_max=1.0):
        if ax1 is None:
            fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(5,5))
        if self.thresh_flag:
            pc_x = self.phasor_coords[~self.binary,0]
            pc_y = self.phasor_coords[~self.binary,1]
        else:
            pc_x = self.phasor_coords[:,:,0]
            pc_y = self.phasor_coords[:,:,1]  
        
        c1=plt.Circle((0,0),radius=1,color='k',fill=True,zorder=-100)
        ax1.add_patch(c1)
        sns.histplot(ax=ax1,x=pc_x.flatten(),y=pc_y.flatten(),cmap=cmap,zorder=-10,bins=100,pmax=p_max)
        pks = self.pc_pks
        ax1.scatter([pc[0] for pc in pks],[pc[1] for pc in pks],c='w',marker='x')
        ax1.plot([pc[0] for pc in self.pc_pks[0:2]],[pc[1] for pc in self.pc_pks[0:2]],c='w',linestyle='--')
        c1=plt.Circle((0,0),radius=1,color='k',fill=True,zorder=-100)
        ax1.add_patch(c1)
        ax1.set_xlim([-1.05,1.05]); ax1.set_ylim([-1.05,1.05]); 
        ax1.set_xticks([]); ax1.set_yticks([]);
        ax1.set_aspect('equal'); 

        ax1.set_xlabel(r'$G$'); ax1.set_ylabel(r'$S$');
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        if plot_save:
            plt.savefig(self.name+'_phasorcloud_pc_peaks.svg',dpi=600)
        return ax1
    
    @staticmethod
    def sign_test(p1,p2,p3):
        return ((p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0] - p3[0])*(p1[1] - p3[1]))

    @staticmethod
    def point_in_tri(pt, v1, v2, v3):
        
        d1 = PairMixture.sign_test(pt,v1,v2)
        d2 = PairMixture.sign_test(pt,v2,v3)
        d3 = PairMixture.sign_test(pt,v3,v1)
        
        has_neg = np.logical_or(np.logical_or(d1<0,d2<0),d3<0)
        has_pos = np.logical_or(np.logical_or(d1>0,d2>0),d3>0)
        
        return np.logical_not(np.logical_and(has_neg,has_pos))
    
    def calc_band(self,n_sigma=1.96,plot=False,cmap=Phasor.cmap,plot_save=False,ax1=None):        
            
        self.dx = self.pc_pks[1][0]-self.pc_pks[0][0]
        self.dy = self.pc_pks[1][1]-self.pc_pks[0][1]
        
        pc_cov = []
        vars_temp = []
        
        unitv = (1/np.sqrt(self.dx**2 + self.dy**2))*(self.dx*np.array([1,0])+self.dy*np.array([0,1]))
          
        self.thresh = n_sigma*(np.amax(np.array([pc.cov for pc in self.purecomps]))/(2*np.sqrt(2)))
        
        vhat_par = self.thresh/(np.sqrt(1+(self.dx/self.dy)**2))*np.array([1, -self.dx/self.dy])

        pvertices = np.array([self.pc_pks[0] + vhat_par , self.pc_pks[1] + vhat_par])
        nvertices = np.array([self.pc_pks[0] - vhat_par , self.pc_pks[1] - vhat_par])
        temp = np.array([pvertices[0],nvertices[0],nvertices[1],pvertices[1]])
        
        self.vertices = temp
        
        data_x = self.phasor_coords[...,0].flatten(); data_y = self.phasor_coords[...,1].flatten()
        self.X = np.stack((data_x, data_y),axis=1)
        
        n = 1000
        xx, yy = np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n))
        X = np.dstack((xx.flatten(),yy.flatten()))[0]
        
        if self.pc_pks.shape[0] > 2:
            
            self.centroid_x = np.sum(np.array([p[0] for p in self.pc_pks]))/3 
            self.centroid_y = np.sum(np.array([p[1] for p in self.pc_pks]))/3 

            self.dxs = np.array([p[0]-self.centroid_x for p in self.pc_pks])
            self.dys = np.array([p[1]-self.centroid_y for p in self.pc_pks])

            unitvs = np.zeros((3,2))
            for i in range(3):
                unitvs[i,:] = (1/np.sqrt(self.dxs[i]**2 + self.dys[i]**2))*(self.dxs[i]*np.array([1,0])+self.dys[i]*np.array([0,1]))
                
            vhats = self.thresh*unitvs
            self.pvertices = np.array([self.pc_pks[i] + vhats[i] for i in range(3)])

            pcoords = self.phasor_coords

            x = pcoords[...,0].flatten()
            y = pcoords[...,1].flatten()
            coords = np.stack((x,y)).T
            self.idx = np.zeros(coords.shape[0])
            for j in range(coords.shape[0]):
                self.idx[j] = PairMixture.point_in_tri(coords[j],self.pvertices[0],self.pvertices[1],self.pvertices[2])
            
            self.idx = self.idx > 0.0
            
        else:

            AB = temp[1]-temp[0]
            BC = temp[2]-temp[1]
            AX = self.X-temp[0]
            BX = self.X-temp[1]
            Adot = np.sum(AB*AX,axis=1)
            Bdot = np.sum(BC*BX,axis=1)
            self.idx = (Adot >=0) & (np.dot(AB,AB) >= Adot) & (Bdot >= 0) & (np.dot(BC,BC) >= Bdot)
        
        self.X_thresh_x = np.ones(np.shape(self.idx)); self.X_thresh_y = np.ones(np.shape(self.idx))
        self.X_thresh_x[self.idx] = np.reshape(self.phasor_coords,np.shape(self.idx)+(2,))[self.idx,0]
        self.X_thresh_y[self.idx] = np.reshape(self.phasor_coords,np.shape(self.idx)+(2,))[self.idx,1]

        if plot:
            if ax1 is None:
                fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
            c1=plt.Circle((0,0),radius=1,color='k',fill=True,zorder=-100)
            ax1.add_patch(c1)
            sns.histplot(ax=ax1,x=self.X_thresh_x[self.X_thresh_x<1].flatten(),
                         y=self.X_thresh_y[self.X_thresh_x<1].flatten(),cmap=cmap,zorder=-10)
            ax1.plot([pc[0] for pc in self.pc_pks[0:2]],[pc[1] for pc in self.pc_pks[0:2]],'w--')

            ax1.set_xlim([-1.0,1.0]); ax1.set_ylim([-1.0,1.0]); ax1.set_aspect('equal'); ax1.set_title(self.name)

            ax1.set_xlabel(r'$G$'); ax1.set_ylabel(r'$S$')
            if plot_save:
                plt.savefig(self.name+'_cluster-defined_band.png',dpi=300)
            return self, ax1
        return self
    
    def plot_band(self,ax1=None,cmap=Phasor.cmap,plot_save=False,p_max=1.0):
        if ax1 is None:
            fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(8,6))

        c1=plt.Circle((0,0),radius=1,color='k',fill=True,zorder=-100)
        ax1.add_patch(c1)

        if self.pc_pks.shape[0] > 2:
            pc_x = self.X_thresh_x[self.idx]
            pc_y = self.X_thresh_y[self.idx]
        else:
            if self.thresh_flag:
                pc_x = self.phasor_coords[~self.binary,0]
                pc_y = self.phasor_coords[~self.binary,1]
            else:
                pc_x = self.phasor_coords[...,0]
                pc_y = self.phasor_coords[...,1]   
        sns.histplot(ax=ax1,x=pc_x.flatten(),y=pc_y.flatten(),cmap=cmap,zorder=0,stat='density',bins=100,pmax=p_max)

        ax1.plot([self.vertices[0,0],self.vertices[3,0]],[self.vertices[0,1],self.vertices[3,1]],c='gray',linestyle='--')
        ax1.plot([self.vertices[1,0],self.vertices[2,0]],[self.vertices[1,1],self.vertices[2,1]],c='gray',linestyle='--')
        ax1.scatter(self.pc_pks[:,0],self.pc_pks[:,1],c='w',marker='o',s=10)
        ax1.set_xlim([-1.05,1.05]); ax1.set_ylim([-1.05,1.05]); ax1.set_aspect('equal'); 
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax1.set_xlabel(r'$G$'); ax1.set_ylabel(r'$S$');
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        if plot_save:
            plt.savefig(self.name+'_cluster-defined_band.svg',dpi=600)
        return self, ax1
        
    @staticmethod
    def frac_helper(A_mat, data_x, data_y, idt, item):

        if idt[item]:
            boo = 1
            fs = lsq_linear(A_mat,[data_x[item],data_y[item],1.0],bounds=(0.0,1.0),lsq_solver='exact')
            res_1 = fs.x[0]
            res_2 = fs.x[1] 
        else:
            boo = 0
            res_1 = 0.0
            res_2 = 0.0

        return res_1, res_2, boo, item
    
    
    def calc_fraction(self,n_sigma=1.96,plot=False,plot_save=False,ax1=None,color='k',density=False,recomp=False,af_ints=None):
        
        if not hasattr(self,'X_thresh_x') or recomp:
            self.calc_band(n_sigma=n_sigma)
        
        if len(self.pc_pks) > 2 :

            self.rescale_af(af_ints)
            v = (self.dx*np.array([1,0])+self.dy*np.array([0,1]))
            p0 = self.pc_pks[0]
            temp_x = np.dstack((self.X_thresh_x,self.X_thresh_y))
            self.x_proj = v*(np.sum(v*(temp_x[0]-p0),axis=1)/np.dot(v,v))[:,None]+p0
        else:
            v = (self.dx*np.array([1,0])+self.dy*np.array([0,1]))
            p0 = self.pc_pks[0]
            temp_x = np.dstack((self.X_thresh_x,self.X_thresh_y))
            self.x_proj = v*(np.sum(v*(temp_x[0]-p0),axis=1)/np.dot(v,v))[:,None]+p0

        tempx = self.x_proj[:,0]
        tempy = self.x_proj[:,1]
        self.fs_tot = np.zeros((len(tempx),2))
        self.ids = np.zeros_like(tempx)
        x1 = [pc[0] for pc in self.pc_pks[0:2]]+[0.0]
        x2 = [pc[1] for pc in self.pc_pks[0:2]]+[0.0]
        x3 = [1.0 for pc in self.pc_pks[0:2]]+[0.0]

        A = [x1, x2, x3]
        temp = np.linalg.lstsq(A,np.stack((tempx,tempy,np.ones(len(tempx)))),rcond=None)
        self.fs_tot[self.idx,0] = temp[0][0][self.idx]
        self.fs_tot[self.idx,1] = temp[0][1][self.idx]
        self.ids_b = self.idx.astype(bool)

        self.fs_norm = np.zeros_like(self.fs_tot[:,0])
        self.fs_norm[self.ids_b] = self.fs_tot[self.ids_b,0]
        self.id2 = self.fs_norm > 0.5
        self.fs_norm[self.id2] = self.fs_tot[self.id2,1]
        self.fs_norm *= 2
        
        self.band = np.dstack((self.X_thresh_x[self.ids_b],self.X_thresh_y[self.ids_b]))[0,:,:]
        pca = PCA(n_components=2)
        model = pca.fit(self.band)
        temp_cov = model.get_covariance()
        v, w = np.linalg.eigh(temp_cov)

        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        self.band_cov = np.amax(v)

        if plot:
            if ax1 is None:
                fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
            if density:
                sns.kdeplot(ax=ax1,data=self.fs_norm[self.ids_b],color=color)
                ax1.set_ylabel(r'Probability Density')
            else:
                ax1.hist(self.fs_tot[self.ids_b,0],bins=50,density=density,c=color);
                ax1.set_ylabel(r'Probability Density')
            ax1.set_xlim([0.0,1.0]); 
            ax1.set_xlabel(r'$f_{mix}$'); 
            if plot_save:
                plt.savefig(self.name+'_fraction_plot.svg',dpi=600)
            return self, ax1
        return self
    
    def plot_fraction(self,ax1=None,density=False):
        if ax1 is None:
            fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
        plt.hist(self.fs_norm[self.ids.astype(bool)],bins=50,density=density,c='k');
        ax1.set_xlim([0.0,1.0]); 
        ax1.set_xlabel(r'$f_{pc1}$'); 
        if density:
            ax1.set_ylabel(r'Probability Density')
        else:
            ax1.set_ylabel(r'# Pixels')
        return self, ax1
    
    @staticmethod
    def frac_helper_b(A_mat, data_x, data_y, item):

        boo = 1
        fs = lsq_linear(A_mat,[data_x[item],data_y[item],1.0],bounds=(0.0,1.0),lsq_solver='exact')
        res = fs.x

        return res, item
    
    def rescale_af(self,plot=False,plot_save=False,ax1=None,color='k',density=False,recomp=False):
        
        x1 = [pc[0] for pc in self.pc_pks]+[0.0]
        x2 = [pc[1] for pc in self.pc_pks]+[0.0]
        x3 = [1.0 for pc in self.pc_pks]+[0.0]
        A = [x1, x2, x3]
        self.res = np.zeros(len(self.X_thresh_x[self.idx]))
        thread_count=int(0.875 * cpu_count())
        with Pool() as pool:
            items = range(len(self.X_thresh_x[self.idx]))
            for result1, jj in pool.map(partial(PairMixture.frac_helper_b,A,self.X_thresh_x[self.idx],self.X_thresh_y[self.idx]),items):
                self.res[jj] = result1[2]
        
        af_ints = np.mean(np.nan_to_num(self.purecomps[-1].intensity),axis=(0,1))
        temp_int = np.zeros((len(self.X_thresh_x),len(af_ints)))
        temp_int[self.idx,:] = self.intensity[np.reshape(self.idx,(self.intensity.shape[0],self.intensity.shape[1]))]
        
        test_af = np.zeros((len(self.X_thresh_x),len(af_ints)))
        
        temp = np.sum(self.intensity,axis=-1).flatten()

        af_temp = np.sum(np.nan_to_num(self.purecomps[-1].intensity),axis=-1)  
        af_thresh = threshold_triangle(af_temp)
         
        af_pixel_int = np.mean(af_temp[af_temp>af_thresh])    

        test_af[self.idx,:] = np.outer(self.res*af_pixel_int,af_ints/np.sum(af_ints))

        testtest = temp_int-test_af
        testtest[testtest < 0.0] = 0.0
        
        self.test_af = test_af
        
        size = np.shape(temp_int)
        lambdas_temp = np.tile(self.lambdas,(size[0],1))

        self.rescaled_phasor_coords = np.nan_to_num(np.moveaxis(Phasor.integrate_(testtest,lambdas_temp,harm=1,ver=0),0,-1))

        self.X_thresh_x_old = np.copy(self.X_thresh_x)
        self.X_thresh_y_old = np.copy(self.X_thresh_y)
        self.idx_old = np.copy(self.idx)

        AB = self.vertices[1]-self.vertices[0]
        BC = self.vertices[2]-self.vertices[1]
        AX = self.rescaled_phasor_coords-self.vertices[0]
        BX = self.rescaled_phasor_coords-self.vertices[1]
        Adot = np.sum(AB*AX,axis=1)
        Bdot = np.sum(BC*BX,axis=1)
        self.idx_1 = (Adot >=0) & (np.dot(AB,AB) >= Adot) & (Bdot >= 0) & (np.dot(BC,BC) >= Bdot)
        
        self.intensity_af_sub = np.reshape(testtest,self.intensity.shape)
        t1 = np.sum(np.nan_to_num(self.intensity_af_sub),axis=-1)
        
        if self.thresh_method is None:
            thresh1 = 0.0
        if self.thresh_method is not None:
            if self.thresh_method == 'otsu':
                thresh1 = threshold_otsu(t1)
            elif self.thresh_method == 'li':
                thresh1 = threshold_li(t1)
            elif self.thresh_method == 'mean':
                thresh1 = threshold_mean(t1)
            elif self.thresh_method == 'triangle':
                thresh1 = threshold_triangle(t1)
            elif self.thresh_method == 'iso':
                thresh1 = threshold_isodata(t1)
        else:

            thresh1 = threshold_otsu(t1)

        mask = t1 >= thresh1
        self.intensity_af_sub[~(mask>0),:] = np.nan
        
        self.idx = np.logical_and(self.idx,np.logical_and(self.idx_1,mask.flatten()))
            
        
        self.X_thresh_x = np.ones(np.shape(self.idx)); self.X_thresh_y = np.ones(np.shape(self.idx))
        self.X_thresh_x[self.idx] = np.reshape(self.rescaled_phasor_coords,np.shape(self.idx)+(2,))[self.idx,0]
        self.X_thresh_y[self.idx] = np.reshape(self.rescaled_phasor_coords,np.shape(self.idx)+(2,))[self.idx,1]
        
        self.rescaled_phasor_coords = np.reshape(self.rescaled_phasor_coords,np.shape(self.phasor_coords))
        
        return self
    
    @staticmethod
    def im_normalize(im):
        im_norm = (im-np.amin(im))/(np.amax(im)-np.amin(im))
        return im_norm
    
    def color_mixing_image(self,plot_save=False,ax1=None,imtype='2D', cmap='magma', quad=False, lims=[2,99]):
        if ax1 is None:
            fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(10,8))
        if self.pc_pks.shape[0] > 2:
            self.img_mean = np.sum(1.0*np.nan_to_num(self.intensity_af_sub),axis=-1)
        else:
            self.img_mean = np.sum(1.0*np.nan_to_num(self.intensity),axis=-1)

        self.img_norm = self.img_mean*np.reshape(self.fs_norm,self.img_mean.shape)
        self.img_norm = PairMixture.im_normalize(self.img_norm)
        plow, phigh = np.percentile(self.img_norm, (lims[0], lims[1]))
        self.img_norm = rescale_intensity(self.img_norm, in_range=(plow, phigh))
        
        self.img_norm_flip = np.flipud(self.img_norm)
        obj=cmap
        cmap = mpl.colormaps.get_cmap(obj)

        if quad:
            self.LUT_2D, self.im_2DLUT = im2DLUT_quad(np.reshape(self.fs_norm,self.img_mean.shape),self.img_mean,
                                cmap=cmap,intLim=np.percentile(self.img_mean,lims))
        else: 
            self.LUT_2D, self.im_2DLUT = im2DLUT(np.reshape(self.fs_norm,self.img_mean.shape),self.img_mean,
                                    cmap=cmap,intLim=np.percentile(self.img_mean,lims))

        if self.img.dims['Z'][0] > 1:
            if imtype != '2D':
                CS = ax1.pcolormesh(np.amax(self.img_norm_flip,axis=0),cmap='magma',vmin=0.0,vmax=1.0)
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes("right", size="10%", pad=0.2)
                
                cbar = plt.colorbar(CS, cax=cax)
                cbar.ax.set_ylabel(r'Mixing Extent',rotation=-90,fontsize=12)
                cbar.ax.set_yticks(np.arange(0,1.1,0.2),['0.0','','','','','1.0'])
            else:
                CS = ax1.pcolormesh(np.flipud(np.amax(self.im_2DLUT,axis=0)))
                divider = make_axes_locatable(ax1)
                ax = divider.append_axes('right',size='25%',pad=0.005)
                ax.imshow(np.flipud(self.LUT_2D))
                ax.set_aspect(10.0)
                ax.set_xticks([])
                ax.set_yticks(np.arange(0,256,51),['1.0','','','','','0.0'])
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_label_text('Mixing Extent',rotation=-90,fontsize=12)
                
        else:
            if imtype != '2D':
                CS = ax1.pcolormesh(self.img_norm_flip,cmap='magma',vmin=0.0,vmax=1.0)
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes("right", size="10%", pad=0.2)
                
                cbar = plt.colorbar(CS, cax=cax)
                cbar.ax.set_ylabel(r'Mixing Extent',rotation=-90,fontsize=12)
                cbar.ax.set_yticks(np.arange(0,1.1,0.2),['0.0','','','','','1.0'])
            else:
                CS = ax1.pcolormesh(np.flipud(self.im_2DLUT))
                divider = make_axes_locatable(ax1)
                ax = divider.append_axes('right',size='25%',pad=0.025)
                ax.imshow(np.flipud(self.LUT_2D))
                ax.set_aspect(10.0)
                ax.set_xticks([])
                ax.set_yticks(np.arange(0,256,51),['1.0','','','','','0.0'])
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_label_text('Mixing Extent',rotation=-90,fontsize=12)
        ax1.set_aspect('equal'); 
        ax1.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False,
            left=False,
            right=False,
            labelleft=False
        )
        # plt.show()
        if plot_save:
            plt.savefig(self.name+'_fraction-normalized_image.svg',dpi=600)
        return ax1
    
    def analyze_overlap(self,n_sigma=1.96,plot=True,plot_save=False,fig=None,axes=None,sim_flag=False,
                        p_max=1.0,lims=[2,98],imtype='2D',cmap='magma', quad=False,recomp=False,af_ints=None):
        if not sim_flag:
            if fig is None:
                fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
            else:
                axes = axes
            self.calc_fraction(n_sigma=n_sigma,density=True,plot=plot,ax1=ax2,recomp=recomp,af_ints=af_ints)
            if plot:
                self.plot_band(ax1=ax1,p_max=p_max)
                
                self.color_mixing_image(ax1=ax3,imtype=imtype,cmap=cmap,quad=quad,lims=lims)
            self.calc_coloc(plot=False)
            print(self.name + ' ' + 'has a colocalization factor of ' + str(self.coloc_vec))
            if plot_save:
                plt.savefig(self.name+'_overlap_analysis.svg',dpi=600)
            return axes
        else:
            if fig is None:
                fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(12,4))
            self.plot_hist_with_peaks(ax1=ax1)
            self.calc_fraction(n_sigma=n_sigma,density=True,plot=plot,ax1=ax2)
            self.calc_coloc(plot=False)
            print(self.name + ' ' + 'has a colocalization factor of ' + str(self.coloc))
            if plot_save:
                plt.savefig('paper_figures/'+self.name+'_overlap_analysis.svg',dpi=300)
            return axes
    
    def calc_coloc(self,plot=False,ax1=None):
        temp = self.fs_norm[self.ids_b]
        h, bins = np.histogram(temp,bins=100,density=True)
        self.coloc = np.trapz(h*bins[1:],bins[1:])
        
        self.pc_cov = np.amax(np.array([pc.cov for pc in self.purecomps]))
        pcs = np.vstack((self.pc_clusters[0],self.pc_clusters[1]))
        pca = PCA(n_components=2)
        model = pca.fit(pcs)
        temp_cov = model.get_covariance()
        v, w = np.linalg.eigh(temp_cov)

        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        self.var_max = np.amax(v)
        self.var_norm = np.clip((self.var_max - self.band_cov)/(self.var_max - self.pc_cov),a_min=0.0,a_max=1.0)
        self.coloc_vec = np.array([self.coloc,self.var_norm])

        if plot:
            if ax1 is None:
                fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
                ax1.stairs(h,bins,c='k')
                ax1.set_xlabel(r'$f_{mix}')
                ax1.set_ylabel(r'Probability Density')
        return self
    
    def calc_pearson_mander(self,pcs,lims=[2,98],thresh_override=False, thresh_method = None):
    
        self.temp_img_pc = np.zeros(np.shape(self.phasor_coords))
        temp_img_pc1 = np.zeros_like(self.temp_img_pc)
        j = 0
        self.lambda_ids = []
        for pc in pcs:
            if type(pc) is str:
                pc_temp = np.loadtxt(pc,delimiter=',',skiprows=1)

                peaks, test = find_peaks(pc_temp[:,-1],height=np.median(pc_temp.flatten()))
                peaks = [peaks[np.argmax(test['peak_heights'])]]
                results_half_pc = peak_widths(pc_temp[:,-1], peaks)
                id_temp = [(self.lambdas>pc_temp[results_half_pc[-2][0].astype(int),0]) & (self.lambdas<pc_temp[results_half_pc[-1][0].astype(int),0])]
                self.lambda_ids.append(id_temp)
            
            else:
                pc_temp = np.zeros(pc.data_cyx.shape[0])
                for i in range(pc.data_cyx.shape[0]):
                    pc_temp[i] = np.mean(np.nan_to_num(pc.intensity[...,i]))                

                peaks, test= find_peaks(pc_temp,height=np.median(pc_temp.flatten()))

                peaks = [peaks[np.argmax(test['peak_heights'])]]
                results_half_pc = peak_widths(pc_temp, peaks)

                id_temp = [(self.lambdas>self.lambdas[results_half_pc[-2][0].astype(int)]) & (self.lambdas<self.lambdas[results_half_pc[-1][0].astype(int)])]
                self.lambda_ids.append(id_temp)

            if self.pc_pks.shape[0] > 2:
                temp1 = np.nan_to_num(self.intensity_af_sub[...,id_temp[0]],nan=0.0,posinf=0.0,neginf=0.0)
            else:
                temp1 = np.nan_to_num(self.intensity_orig[...,id_temp[0]],nan=0.0,posinf=0.0,neginf=0.0)

            im_pc = np.sum(temp1,axis=-1)

            p2, p98 = np.percentile(im_pc, (lims[0], lims[1]))
            im_pc = rescale_intensity(im_pc, in_range=(p2, p98))
            temp_im = np.copy(im_pc)
            
            self.temp_img_pc[...,j] = im_pc[...]
            temp_img_pc1[...,j] = temp_im[...]
            j += 1
        
        
        t1 = np.copy(self.temp_img_pc[...,0])
        t2 = np.copy(self.temp_img_pc[...,1])
        t3 = np.copy(self.temp_img_pc[...,1])
        if self.thresh_method is not None:
            if self.thresh_method == 'otsu':
                thresh1 = threshold_otsu(t1)
                thresh2 = threshold_otsu(t2)
            elif self.thresh_method == 'li':
                thresh1 = threshold_li(t1)
                thresh2 = threshold_li(t2)
            elif self.thresh_method == 'mean':
                thresh1 = threshold_mean(t1)
                thresh2 = threshold_mean(t2)
            elif self.thresh_method == 'triangle':
                thresh1 = threshold_triangle(t1)
                thresh2 = threshold_triangle(t2)
            elif self.thresh_method == 'iso':
                thresh1 = threshold_isodata(t1)
                thresh2 = threshold_isodata(t2)
            else:
                print('Threshold method not regognized. Using Otsu')
                thresh1 = threshold_otsu(t1)
                thresh2 = threshold_otsu(t2)
        elif thresh_override:
            if thresh_method == 'otsu':
                thresh1 = threshold_otsu(t1)
                thresh2 = threshold_otsu(t2)
            elif thresh_method == 'li':
                thresh1 = threshold_li(t1)
                thresh2 = threshold_li(t2)
            elif thresh_method == 'mean':
                thresh1 = threshold_mean(t1)
                thresh2 = threshold_mean(t2)
            elif thresh_method == 'triangle':
                thresh1 = threshold_triangle(t1)
                thresh2 = threshold_triangle(t2)
            elif thresh_method == 'iso':
                thresh1 = threshold_isodata(t1)
                thresh2 = threshold_isodata(t2)
            else:
                print('Threshold method not given by requested. Using Otsu')
                thresh1 = threshold_otsu(t1)
                thresh2 = threshold_otsu(t2)   
        else:
            thresh1 = 0.0
            thresh2 = 0.0
        
        mask1 = t3 > thresh2
        mask2 = t1 > thresh1
        
        self.p_mask = np.logical_and(mask1,mask2)
        self.pearsons = pearson_corr_coeff(t1,t2,mask=np.logical_and(mask1,mask2)).statistic
        
        self.mandersO = manders_overlap_coeff(t1,t2,mask=np.logical_or(mask1,mask2))
        
        self.test_mask = np.logical_and(mask1,mask2)
        
        self.manders1 = manders_coloc_coeff(t1,mask1,mask2)
        self.manders2 = manders_coloc_coeff(t2,mask2,mask1)
        
        self.mask1 = mask1
        self.mask2 = mask2

        return self
    
    def save_data(self,path):
        with open(path,'wb') as f:
            pickle.dump(self,f)
        
    