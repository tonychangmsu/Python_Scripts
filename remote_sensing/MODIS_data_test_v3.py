#Title: MODIS_data_test.py
#Author: Tony Chang
#Abstract: Test for opening MODIS data and examining the various bands
#Creation Date: 04/14/2015
#Modified Dates: 01/20/2016, 01/26/2016, 01/28/2016, 01/29/2016, 02/01/2016

#local directory : K:\\NASA_data\\scripts

import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("K:\\NASA_data\\scripts")
import time
import MODIS_acquire as moda
import MODIS_tassel_cap as tas
import MODIS_process as mproc
import tiff_write as tw

#MODIS file name as 
# 7 char (product name .)
# 8 char (A YYYYDDD .)
# 6 char (h XX v YY .) #tile index
# 3 char (collection version .) #typically 005
# 14 char (julian date of production YYYYDDDHHMMSS)

if __name__ == "__main__":
	start = time.time()
	#since we have the date, let's try to get all the data from that date together.
	htile = 9
	vtile = 4
	factor = 0.0001
	year = 2000
	#we would iterate through the year
	begin_year = 2000
	end_year = 2015
	wd = 'G:\\NASA_remote_data\\MOD09A1'
	mod_list, mod_dates = moda.mod_file_search(wd, year, True)
	#then iterate through theses list values
	scene = 0 
	mod_data, dnames = moda.mod_acquire_by_file(mod_list[scene]) #this is the full dataset
	band_query = 1
	#get the files needed
	files_to_mosaic = moda.mod_date_dataset_list(wd, mod_dates[scene]) 
	nonproj_mosaics = mproc.mosaic_files(files_to_mosaic, reproj = False)
	reproj_mosaics = mproc.mosaic_files(files_to_mosaic, reproj = True, method = 0)
	#inspect the cloud effects on the nonproj and reproj mosaics
	#looks like it comes from band 5! 1230-1250, ,Leaf/Canopy Differences
	#not much can be done about that if this is prevalent. In the mean time, we should just implement
	#the processing and use the QC to fix the problem
	#at this point we would like to transform the data. Then we can apply the reprojection
	#need to be careful here, do we reproject before transform or after? before...
	transformed = tas.tassel_cap_transform(nonproj_mosaics[:7]) #don't want to include the qc data
	#check out the tasseled_cap again. getting some striping for some reason.
	tw.tiff_write_gdal(transformed[0], 'K:\\NASA_data\\test\\test_clip.tif')
	
	tw.tiff_write(out, x_size, y_size, cell_size, ymax, xmin, 'K:\\NASA_data\\test\\test_clip.tif')
	
	#tas_array = moda.datasets_to_array(transformed, False)
	#find the bounding box by the netCDF from TOPOWX

	#GYE coordinates
	xmin = -112.39583333837999 #112 23 45
	xmax = -108.19583334006 #108 11 45
	ymin = 42.279166659379996 #42 16 45
	ymax = 46.195833324479999 #46 11 45
	aoa = [xmin, xmax, ymin, ymax]

	clip = mproc.clip_wgs84_scene(aoa, transformed[0])
	#some problems with the reprojection process? 
	#NO..getting some strange stripe artifacts from the tasselled cap, but could be inherant in the MOD09 data itself...
	
	#all this works now. So now perform this for all the MODIS data and store it in a netCDF4 file that
	#is continuous for each year.
	
	#write the file to check it out
	tw.tiff_write(clip, np.shape(clip)[1], np.shape(clip)[0], cell_size, ymax, xmin, 'K:\\NASA_data\\test\\', 'test_clip.tif')


	#now just write this function for netCDF4
	#then save to a netCDF4 file
	#then repeat for all the data.
	end = time.time()
	print('run time :%s'%(end-start)) #takes about 25-30 seconds

	'''
	mproc.plot_refl(mod_array) 
	#plot all the reflectances
	#see which is faster
	import time
	start = time.time()
	b,g,w = tas.tassel_cap_transform(mod_array)
	end = time.time()
	mproc.plot_tassel_cap(b,g,w)
	'''
