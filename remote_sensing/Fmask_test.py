'''
Title: 		Fmask_test.py
Date: 		Created 2016.02.16
			Modified
Author: 	Tony Chang
Abstract: 	Tests the Function Mask from Zhu and Woodcock 2012 to mask cloud and cloud shadows from 
			Landsat scenes given the Top of Atmosphere (TOA) reflectances for bands 1,2,3,4,5,7 and Band 6
			Brightness Temperature. For Landsat L1T images, digital number (DN) values are converted to TOA
			reflectances and BT 
			
			Zhu and Woodcock use the LEDAPS system
			Masek, J.G., E.F. Vermote, N. Saleous, R. Wolfe, F.G. Hall, F. Huemmrich, F. Gao, J. Kutler, and T.K. Lim. 2013. LEDAPS Calibration, Reflectance, Atmospheric Correction Preprocessing Code, Version 2. Model product. Available on-line [http://daac.ornl.gov] from Oak Ridge National Laboratory Distributed Active Archive Center, Oak Ridge, Tennessee, U.S.A. http://dx.doi.org/10.3334/ORNLDAAC/1146
			
			Translated from matlab code produced by Zhu found at https://github.com/prs021
			
			
'''
#==================================MAIN================================================
import sys
sys.path.append(r'K:\NASA_data\scripts') #add directory for geotools_LT script
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal as gdal
import os
import geotool_LT as gt
import LANDSAT_tools as landtools
from skimage.measure import label
from skimage.measure import regionprops
#import Fmask as fmask
sys.path.append(r'K:\NASA_data\fmask') #add directory for geotools_LT script
from nd2toarbt import *
from plcloud import *

if __name__ == '__main__':
	pi = 38 #range from 37 to 39
	ri = 29 #range from 28 to 30
	p = np.arange(37,40)
	r = np.arange(28,31)
	begin_year = 2000 #this can decrease to the year 1997...
	end_year = 2001 #work with a single year so we can test out the masker
	years = np.arange(begin_year, end_year)
	#use zip here
	#each tile has a bulk order directory
	band = 1
	all_tiles = []
	r_dirs = []
	all_dirs = []
	for year in range(begin_year, end_year):
		for pi in p[:3]:
			tiles = []
			for ri in r[:3]:
				directory_name = r'p%sr%s'%(pi,ri)
				wd = r'G:\NASA_remote_data\LT5_LEDAPS_processed\%s'%(directory_name)
				scene_dirs = os.listdir(wd) #list of all the landsat scenes
				#just get the first one for now
				dir_list, tif_list = landtools.get_data_list(wd, year)
				scene = dir_list[4]
				metadata = landtools.get_metadata(scene)
				#get the surface reflectance
				bands = landtools.get_bands(scene, 'toa')
				xml_data = landtools.get_xml_metadata(scene)
	
	zen,azi,ptm,Temp,t_templ,t_temph,WT,Snow,Cloud,Shadow,dim,ul,resolu,zc = plcloud(scene,num_Lst = 4)
	
##################
#fcssm test
#get the Cloud layer as the test

#cloud_test = Cloud
#cloud_test = np.array([[2,1,0,0,1],[1,1,0,0,0],[0,0,0,0,1],[1,0,0,0,1],[1,1,1,0,0]])
cloud_test = Cloud
cloud_test[cloud_test!=1] = 0
segm_cloud_init = label(cloud_test, neighbors = 8)
L = segm_cloud_init.astype('uint32')
s = regionprops(L)
area = np.array([i.area for i in s]) #get the area of each labeled cloud
num_cldoj = 3
idx  = np.where(area >= num_cldoj)[0]+1

L_reshape = np.reshape(L, (np.shape(L)[0]*np.shape(L)[1]))
sort_inx = np.argsort(L_reshape)
to_keep_left = np.searchsorted(L_reshape[sort_inx], idx, side = 'left')
to_keep_right = np.searchsorted(L_reshape[sort_inx], idx, side = 'right')
for i in range(len(to_keep_left)):
	L_reshape[sort_inx[to_keep_left[i]:to_keep_right[i]]] = 1
L_reshape[L_reshape!=1] = 0
segm_cloud_tmp = np.reshape(L_reshape,np.shape(L))
segm_cloud = label(segm_cloud_tmp, neighbors = 8)

s = regionprops(segm_cloud)
num = len(s)
area_final = [i.area for i in s]
obj_num = area_final