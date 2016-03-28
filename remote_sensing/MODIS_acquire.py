#Title: MODIS_acquire.py
#Author: Tony Chang
#Abstract: Function for opening MODIS data and examining the various bands
#Modified Dates: 01/20/2016

#local directory : K:\\NASA_data\\scripts

import numpy as np
import matplotlib.pyplot as plt
from pyhdf import SD
import gdal as gdal
import osr
from osgeo import gdal
from osgeo import osr
from mpl_toolkits.basemap import Basemap
import os

def mod_file_search(wd, year, return_dates=False):
#input a directory and year
#returns the files
	mod_list = []
	dates = []
	jdates = []
	y = "%s."%year
	sub_dir = [s for s in os.listdir(wd) if y in s]
	for sub in sub_dir:
		fpath = "%s\\%s"%(wd,sub)
		if return_dates:
			dates.append(sub.split('.'))
		for mod_file in os.listdir(fpath):
			if mod_file.endswith(".hdf"):
				mod_list.append("%s\\%s"%(fpath,mod_file))
				if return_dates:
					jdates.append(mod_file[9:16])
	if return_dates:
		return(mod_list, np.array(dates).astype(int), np.unique(np.array(jdates)))
	else:
		return(mod_list)

def mod_date_dataset_list(wd, date):
	#input a date in list format [year, month, day]
	#returns a list of filenames with the same requested date
	mod_list = []
	date_query = "%04d.%02d.%02d" %(date[0], date[1], date[2])
	sub_dir = [s for s in os.listdir(wd) if date_query in s]
	for sub in sub_dir:
		fpath = "%s\\%s"%(wd,sub)
		for mod_file in os.listdir(fpath):
			if mod_file.endswith(".hdf"):
				mod_list.append("%s\\%s"%(fpath,mod_file))
	return(mod_list)

def mod_acquire_by_file(fname):
	#input fname 
	#returns the individual bands as an array
	hdf = gdal.Open(fname)
	sds = hdf.GetSubDatasets()
	dnames = np.array(sds)[:,0]
	return(hdf, dnames)

def mod_acquire_by_band(fname, band_name):
	#input fname and band name 
	#returns the individual band as a gdal Dataset
	if str(band_name).isdigit():
		band_name = 'b%s'%(str(band_name).zfill(2))
	hdf = gdal.Open(fname)
	sds = hdf.GetSubDatasets()
	dnames = np.array(sds)[:,0]
	query_value = [s for s in dnames if band_name in s][0]
	return(gdal.Open(query_value))

def get_bands(fname, float_type = False):
	#returns the individual 1-7 bands from MOD09 in int16 unless float_type = True
	mod_bands = []
	for i in range(1,8):
		mod_bands.append(mod_acquire_by_band(fname, i).ReadAsArray())
	if float_type:
		out = (np.array(mod_bands)/10000)
	else:
		out = np.array(mod_bands)
	return(out)

def get_geo_info(src):
	geo_t = src.GetGeoTransform()
	proj = osr.SpatialReference()
	proj.ImportFromWkt(src.GetProjectionRef())
	x_size = src.RasterXSize # Raster xsize
	y_size = src.RasterYSize # Raster ysize
	dtype = src.GetRasterBand(1).DataType
	return(geo_t, proj, x_size, y_size, dtype)

def create_gdal_dataset(ref, data_array):
	'''
	input a reference file for spatial data
	outputs a new single band gdal dataset with same spatial attributes
	'''
	gt, prj, x_size, y_size, datatype = get_geo_info(ref)
	#create in-memory raster
	mem_drv = gdal.GetDriverByName('MEM')
	dest = mem_drv.Create('', x_size, y_size, 1, datatype)
	# Calculate the new geotransform
	# Set the geotransform
	dest.SetGeoTransform(gt)
	dest.SetProjection(prj.ExportToWkt())
	#perform the tasseled cap transform
	dest.GetRasterBand(1).WriteArray(data_array)
	return(dest)

def datasets_to_array(datasets, factor_correction=False):
	''' takes in array of gdal datasets and returns the array data'''
	if factor_correction:
		factor = 0.0001
	else:
		factor = 1
	return(np.array([i.ReadAsArray()*factor for i in datasets]))