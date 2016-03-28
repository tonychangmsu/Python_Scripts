#Title: MODIS_data_test.py
#Author: Tony Chang
#Abstract: Test for opening MODIS data and examining the various bands
#Creation Date: 04/14/2015
#Modified Dates: 01/20/2016

#local directory : K:\\NASA_data\\scripts
#%load_ext autoreload
#%autoreload 2

'''
import os
os.chdir("K:\\NASA_data\\scripts")
%load_ext autoreload
%autoreload 2
'''

import numpy as np
import matplotlib.pyplot as plt
import MODIS_acquire as moda
from pyhdf import SD

if __name__ == "__main__":
	mod_wd = "G:\\NASA_remote_data\\MOD09A1"
	year = 2000
	
	m_files, dates = moda.mod_file_search(mod_wd, year, return_dates=True)
	#let's acquire the first data file
	h = 9
	v = 4
	modis, dnames = mdoa.mod_acquire(dates[0][0], dates[0][1],dates[0][2], h, v)
	
	date = "%04d.%02d.%02d\\" %(2000,2,18)
	rs_type = "MOD09A1"
	#in the case of my data I only have two tiles h=[9, 10] and v = 4
	h = "%02d" %(9)
	v = "%02d" %(4)

	#MODIS file name as 
	# 7 char (product name .)
	# 8 char (A YYYYDDD .)
	# 6 char (h XX v YY .) #tile index
	# 3 char (collection version .) #typically 005
	# 14 char (julian date of production YYYYDDDHHMMSS)
	modis = "%s.A2000049.h%sv%s.005.2006268184318.%s" %(rs_type, h, v, "hdf")
	data_file = "%s%s%s" %(mod_wd, date, modis)

	hdf = SD.SD(data_file)
	datasets = hdf.datasets()
	dnames = list(datasets.keys()) #list of all the keys in the MODIS dataset
	query = 'qc' 
	#query = 'day'
	#if checking for the quality control, know which bands you are interested in being correct because qc returns 
	#an integer that must be converted back into 32bit unsigned binary for qc purposes
	matching = [s for s in dnames if query in s ] #search the list for the query
	band = matching[0]
	sds = hdf.select(band)
	data = sds.get()
	print(np.binary_repr(data.max()))
	print(np.binary_repr(data.min()))
	#or the binary representation of all the data
	#out = np.array(np.binary_rep(s) for s in data.flatten()[0]).reshape(data.shape)
	#test for qc first? see page 22 of http://www.gscloud.cn/userfiles/file/MOD09_UserGuide.pdf
	#selecting qc
	
	#now repeat this process for all the dates that are in the dataset

	#try with some reflectance band (band 1 = red), (band 2 = NIR), (band 3= blue)
	#we need brightness, wetness (band 6 and 7 - veg moisture/soil moisture middle IR), and greeness (band 1 and 2)
	query = '01'
	matching = [s for s in dnames if query in s ] #search the list for the query
	band = matching[0]
	sds = hdf.select(band)
	data = sds.get()
	f, ax = plt.subplots(nrows = 1, ncols =1, figsize=(8, 6), dpi=80)
	ds_img = ax.imshow(data, cmap ='viridis')
	cbar = f.colorbar(ds_img)

	#now try to mosaic?
	#since we have the date, let's try to get all the data from that date together.
	
