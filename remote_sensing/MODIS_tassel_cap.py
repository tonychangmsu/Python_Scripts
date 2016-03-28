#Title: MODIS_tassel_cap.py
#Author: Tony Chang
#Abstract: Function to convert MOD09A1 data the Tasseled Cap data spaces for brightness, greenness and wetness based on Zhang et al 2002
#Creation Dates: 2016.01.20
#Modified Dates: 2016.01.21,

#local directory : K:\NASA_data\scripts
#%load_ext autoreload
#%autoreload 2
'''
import os
os.chdir("K:\\NASA_data\\scripts")
%load_ext autoreload
%autoreload 2
'''
import numpy as np
import MODIS_acquire as moda
from osgeo import gdal, osr
import matplotlib.pyplot as plt

"""Parameters for the Tasseled Cap tranformation of MOD09 data"""

##############################################################################
MBRIGHT = np.array([0.3956, 0.4718, 0.3354, 0.3834, 0.3946, 0.3434, 0.2964])
MGREEN = np.array([-0.3399, 0.5952, -0.2129, -0.2222, 0.4617, -0.1037, -0.46])
MWET = np.array([0.10839, 0.0912, 0.5065, 0.404, -0.241, -0.4658, -0.5306])
##############################################################################

#ideally construct a class for this data.

def tassel_cap_transform(mod_data, br = MBRIGHT, gr = MGREEN, we = MWET):
	'''
	input the 7 bands of the MODIS data in grid form
	returns the tassel cap transforms for a specified MODIS scene
	finds each band within the HDF dataset using a tensordot product.
	
	using the broadcast method is more clear, but twice the time to process.
	bright = np.sum(np.array(data_array)* br[:,None,None],axis =0) #broadcast the parameters
	green = np.sum(np.array(data_array)* gr[:,None,None],axis =0) #broadcast the parameters
	wet = np.sum(np.array(data_array)* we[:,None,None],axis =0) #broadcast the parameters
	'''
	ref_band = mod_data[0]
	#perform the tasseled cap transform
	mod_data_array = moda.datasets_to_array(mod_data, factor_correction = True)
	tc = np.tensordot(mod_data_array,np.array([br,gr,we]), axes=(0,1))
	br_out = moda.create_gdal_dataset(ref_band, tc[:,:,0])
	gr_out = moda.create_gdal_dataset(ref_band, tc[:,:,1])
	we_out = moda.create_gdal_dataset(ref_band, tc[:,:,2])
	return(br_out, gr_out, we_out)

def ndviTransform(mod_data):
	band4 = band_query(mod_data, 4)
	band3 = band_query(mod_data, 3) 
	ndvi = (band4-band3)/(band4+band3)
	return(ndvi)