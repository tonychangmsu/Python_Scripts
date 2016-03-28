#Title: MODIS_resample
#Author: Tony Chang
#Date: 4.16.2015
#Abstract: Resamples the MODIS data from a 500m pixel to 30m (match with Landsat)
#

import osr
import gdal 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pyhdf.SD import SD, SDC not required, just use GDAL

def reproject_dataset ( dataset, pixel_spacing=5000., epsg_from=4326, epsg_to=4326 ):
	"""
		A function to resample a GDAL dataset from within Python. The idea
		is to change the pixel size by calculating the number of pixels desired.
	"""
if __name__ == '__main__': 
	mod_wd = "K:\\NASA_data\\MOD09A1\\"
	date = "%04d.%02d.%02d\\" %(2000,2,18)
	rs_type = "MOD09A1"
	iv = 4
	ih = 9
	h = "%02d" %(ih)
	v = "%02d" %(iv)
	
	#MODIS file name as 
	# 7 char (product name .)
	# 8 char (A YYYYDDD .)
	# 6 char (h XX v YY .) #tile index
	# 3 char (collection version .) #typically 005
	# 14 char (julian date of production YYYYDDDHHMMSS)

	modis = "%s.A2000049.h%sv%s.005.2006268184318.%s" %(rs_type, h, v, "hdf")
	data_file = "%s%s%s" %(mod_wd, date, modis)

	ds = gdal.Open(data_file, gdal.GA_ReadOnly)
	md = ds.GetMetadata_Dict()
	#define bounding box as [minx, maxx, miny, maxy]
	bbx_txt = [md['WESTBOUNDINGCOORDINATE'], md['EASTBOUNDINGCOORDINATE'], \
		md['SOUTHBOUNDINGCOORDINATE'], md['NORTHBOUNDINGCOORDINATE']]
	bbx = [float(i) for i in bbx_txt] #change to float
	
	sub_ds = ds.GetSubDatasets()
	#okay now we can get individual raster bands using gdal
'''
	src = gdal.Open(sub_ds[0][0], gdal.GA_ReadOnly)
	src_gtrn =  src.GetGeoTransform()
	src_bbox_cells = ((0., 0.), (0, src.RasterYSize), (src.RasterXSize, 0),(src.RasterXSize, src.RasterYSize))
	geo_pts_x = []
	geo_pts_y = []
	for x, y in src_bbox_cells:
		x2 = src_gtrn[0] + src_gtrn[1] * x + src_gtrn[2] * y
		y2 = src_gtrn[3] + src_gtrn[4] * x + src_gtrn[5] * y
		geo_pts_x.append(x2)
		geo_pts_y.append(y2)
	bbox = [(min(geo_pts_x), min(geo_pts_y), max(geo_pts_x), max(geo_pts_y)]
'''	
	
	red  = band1.GetRasterBand(1).ReadAsArray() #indexing starts at 1!
	plt.imshow(red, cmap ='Reds', extent = bbx);plt.colorbar()
	#this plot looks funny, because everything is all skewed over
	#what we would like to do is project this to WGS84
	#first thing is to write a generic function
	
	ds = gdal.Open(data_file, gdal.GA_ReadOnly)
	sub_ds = ds.GetSubDatasets()
	g = gdal.Open(sub_ds[0][0])
	rd = g.GetRasterBand(1).ReadAsArray()
	geo_transform = g.GetGeoTransform()
	x_size = g.RasterXSize # Raster xsize
	y_size = g.RasterYSize # Raster ysize
	srs = g.GetProjectionRef() # Projection
	# Need a driver object. By default, we use GeoTIFF
	driver = gdal.GetDriverByName('GTiff')
	output_name = r'K:\Nasa_data\test\test.tif'
	dataset_out = driver.Create(output_name, x_size, y_size, 1, gdal.GDT_Float32)
	dataset_out.SetGeoTransform(geo_transform)
	dataset_out.SetProjection(srs)
	outband = dataset_out.GetRasterBand(1)
	outband.WriteArray(rd.astype(np.float32))
	dataset_out = None
	outband.FlushCache()
	#okay, so this works to save the data as a GeoTIFF
	
	#define the bounding box from <http://modis-land.gsfc.nasa.gov/pdf/sn_bound_10deg.txt>
	#Pixel Size = (463.312716527916677,-463.312716527916507)
	#hdf = SD(data_file, SDC.READ)
	bound_box_list = pd.read_csv(r'K:\NASA_data\sn_bound_10deg.txt', delim_whitespace=True, header=6, skiprows=[655,656])
	bbx = bound_box[(bound_box_list.iv == iv) & (bound_box_list.ih == ih)]
	datasets = hdf.datasets()
	dnames = list(datasets.keys()) #list of all the keys in the MODIS dataset
	query = 'qc' 
	query = 'day'
	#if checking for the quality control, know which bands you are interested in being correct because qc returns 
	#an integer that must be converted back into 32bit unsigned binary for qc purposes
	matching = [s for s in dnames if query in s ] #search the list for the query
	band = matching[0]
	sds = hdf.select(band)
	data = sds.get()
	print(format(data.max(), 'b'))
	print(format(data.min(), 'b'))
	
	#try with some reflectance band (band 1 = red), (band 2 = NIR), (band 3= blue)
	#we need brightness, wetness (band 6 and 7 - veg moisture/soil moisture middle IR), and greeness (band 1 and 2)
	query = '01'
	matching = [s for s in dnames if query in s ] #search the list for the query
	band = matching[0]
	sds = hdf.select(band)
	data = sds.get()
	plt.imshow(data, cmap ='Reds'); plt.colorbar()
	
	#so this works. Now we need to resample it, given the geotransform information
	wgs84 = osr.SpatialReference()
	wgs84.ImportFromEPSG(