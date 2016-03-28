#Title: MODIS_data_test.py
#Author: Tony Chang
#Abstract: Test for opening MODIS data and examining the various bands
#Creation Date: 04/14/2015
#Modified Dates: 01/20/2016, 01/26/2016, 01/28/2016

#local directory : K:\\NASA_data\\scripts

import numpy as np
import matplotlib.pyplot as plt
import MODIS_acquire as moda
import MODIS_tassel_cap as tas
import MODIS_plot as mplot
import tiff_write as tw
import os
import time
#MODIS file name as 
# 7 char (product name .)
# 8 char (A YYYYDDD .)
# 6 char (h XX v YY .) #tile index
# 3 char (collection version .) #typically 005
# 14 char (julian date of production YYYYDDDHHMMSS)

if __name__ == "__main__":
	os.chdir("K:\\NASA_data\\scripts")
	start = time.time()
	#since we have the date, let's try to get all the data from that date together.
	htile = 9
	vtile = 4
	year = 2000
	wd = 'G:\\NASA_remote_data\\MOD09A1'
	mod_list, mod_dates = moda.mod_file_search(wd, year, True)
	mod_data, dnames = moda.mod_acquire_by_file(mod_list[0]) #this is the full dataset
	mband = moda.mod_acquire_by_band(mod_list[0], 1) #this is the single band data
	#get bands 1-7
	mod_array = moda.get_bands(mod_list[0], float_type = True) #get the float values
	re_mband = mplot.reproj_WGS84(mband) #try reprojection function
	band1 = [re_mband.ReadAsArray()/10000]
	'''
	mplot.plot_refl(band1)
	#now try to mosaic the two scenes
	
	#takes in a list of gdal dataset to mosaic together
	ds1 = moda.mod_acquire_by_band(mod_list[0],1)
	ds2 = moda.mod_acquire_by_band(mod_list[1],1)
	mod_mosaic = mplot.mosaic(ds1, ds2)
	msic = [mod_mosaic.ReadAsArray()/10000]
	mplot.plot_refl(msic)
	re_msic = mplot.reproj_WGS84(mod_mosaic) #try reprojection function
	mos_band1 = [re_msic.ReadAsArray()/10000]
	mplot.plot_refl(mos_band1)
	'''
	#looks good
	#so make a function to search through all the bands and mosaic them
	#now perform this for all the datasets, and then clip the data to the sub-set area
	#save the date, and put it all into a netCDF4 file.
	#then I will have all the data put together in a neat package.
	#684 dates, with 2 tiles within each 91 GB in size
	
	#step one get the files needed
	i = 0 #date iterator
	files_to_mosaic = moda.mod_date_dataset_list(wd, mod_dates[i])
	n_files = len(files_to_mosaic)
	#step two open each file and get the bands
	j = 0 #file list iterator
	mod_bands = [[] for i in range(n_files)]
	
	#make a band list
	band_list = list(range(1,8))
	band_list.append('qc') #add the quality control band
	for j in range(n_files):
		for k in band_list:
			mod_bands[j].append(moda.mod_acquire_by_band(files_to_mosaic[j], k))

	#now mosaic each band
	#mosaics = []
	reproj_mosaics = []
	for band in range(len(mod_bands[0])):
		ds = []
		for tile in range(n_files):
			ds.append(mod_bands[tile][band])
		mod_mosaic = mplot.mosaic(ds[0], ds[1])
		#now transform the mosaics
		reproj_mosaics.append(mplot.reproj_WGS84(mod_mosaic))
		#mosaics.append(mod_mosaic.ReadAsArray()/10000)
	#mosaics = np.array(mosaics)
	#check how it looks
	d_check = reproj_mosaics[0].ReadAsArray()
	plt.imshow(np.ma.masked_where(d_check<=0, d_check))
	plt.colorbar(orientation='horizontal')
	#looks good, now clip it
	#find the bounding box by the netCDF from TOPOWX
	
	xmin = -112.39583333837999 #112 23 45
	xmax = -108.19583334006 #108 11 45
	ymin = 42.279166659379996 #42 16 45
	ymax = 46.195833324479999 #46 11 45
	AOA = [xmin, xmax, ymin, ymax]
	test_mosaic = reproj_mosaics[0]
	geo_trans = test_mosaic.GetGeoTransform()
	nx = test_mosaic.RasterXSize
	ny = test_mosaic.RasterYSize
	cell_size = geo_trans[1]
	xul = geo_trans[0] #-140.015144
	yul = geo_trans[3] #49.9999999
	xlr = xul + (cell_size * nx) #-91.390271
	ylr = yul - (cell_size * ny) #40.01671667
	xmin_i = (AOA[0] - xul)/cell_size
	xmax_i = (AOA[1] - xul)/cell_size
	ymin_i = (yul - AOA[2] )/cell_size
	ymax_i = (yul - AOA[3])/cell_size
	i_s = np.array([xmin_i, xmax_i, ymin_i, ymax_i])
	print('%s, %s, %s, %s' %(i_s[0],i_s[1],i_s[2],i_s[3]))
	i_s = np.array([np.floor(xmin_i), np.ceil(xmax_i), np.ceil(ymin_i), np.floor(ymax_i)]).astype(int)
	print('%s, %s, %s, %s' %(i_s[0],i_s[1],i_s[2],i_s[3]))
	#so we just have to subset by the AOA GYE
	clip = d_check[i_s[3]:i_s[2], i_s[0]:i_s[1]]
	#write the file to check it out
	tw.tiff_write(clip, np.shape(clip)[1], np.shape(clip)[0], cell_size, ymax, xmin, 'K:\\NASA_data\\test\\', 'test_clip.tif')
	#now just write this function for netCDF4
	
	#then save to a netCDF4 file
	#then repeat for all the data.
	end = time.time()
	print('run time :%s'%(end-start)) #takes about 25-30 seconds
	
	'''
	mplot.plot_refl(mod_array) 
	#plot all the reflectances
	#see which is faster
	import time
	start = time.time()
	b,g,w = tas.tassel_cap_transform(mod_array)
	end = time.time()
	mplot.plot_tassel_cap(b,g,w)
	'''
