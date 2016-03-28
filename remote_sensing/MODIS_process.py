#Title: MODIS_process.py
#Author: Tony Chang
#Abstract: Function for plotting, reprojecting, and mosaicking MODIS data and examining the various bands
#Created Date: 01/26/2016

#local directory : K:\\NASA_data\\scripts
#%load_ext autoreload
#%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import gdal as gdal
import osr
from osgeo import gdal
from osgeo import osr
from mpl_toolkits.basemap import Basemap

import MODIS_acquire as moda
import MODIS_tassel_cap as tas

def plot_refl(mod_data, factor = 1, nbands = 1):
	#plot a single band of MODIS data
	f, ax = plt.subplots(nrows = 1, ncols = nbands, figsize=(16,16), dpi =300)
	if nbands == 1:
		ds_i = mod_data.ReadAsArray()*factor
		ds_img = ax.imshow(np.ma.masked_where(ds_i <0, ds_i), cmap ='viridis')
		cbar = f.colorbar(ds_img, ax = ax,fraction=0.046, pad=0.04, orientation = 'horizontal')
	else:
		for i in range(nbands):
			ds_i = mod_data[i].ReadAsArray()*factor
			ds_img = ax[i].imshow(np.ma.masked_where(ds_i <0, ds_i), cmap ='viridis')
			cbar = f.colorbar(ds_img, ax = ax[i], fraction=0.046, pad=0.04, orientation = 'horizontal')
	plt.tight_layout()
	plt.show()

def plot_tassel_cap(b,g,w):
	f, ax = plt.subplots(nrows = 1, ncols =3, figsize=(10, 16), dpi=300)
	ds_imgb = ax[0].imshow(b, cmap ='viridis')
	cbar = f.colorbar(ds_imgb, ax = ax[0],fraction=0.046, pad=0.04)
	ax[0].set_title('Brightness Index')
	ds_imgr = ax[1].imshow(g, cmap ='viridis')
	cbar = f.colorbar(ds_imgr, ax = ax[1],fraction=0.046, pad=0.04)
	ax[1].set_title('Greenness Index')
	ds_imgw = ax[2].imshow(w, cmap ='viridis')
	cbar = f.colorbar(ds_imgw, ax = ax[2],fraction=0.046, pad=0.04)
	ax[2].set_title('Wetness Index')
	plt.tight_layout()
	plt.show()

def reproj_wgs84(mod_data, cell_size = 0.004166666666666667, method =  0):
	'''
	#input the SubDataset data in gdal.Dataset format
	/*! Nearest neighbour (select on one input pixel) 	*/ GRA_NearestNeighbour = 0,
	/*! Bilinear (2x2 kernel) 							*/ GRA_Bilinear = 1,
	/*! Cubic Convolution Approximation (4x4 kernel) 	*/ GRA_Cubic = 2,
	/*! Cubic B-Spline Approximation (4x4 kernel) 		*/ GRA_CubicSpline = 3,
	/*! Lanczos windowed sinc interpolation (6x6 kernel)*/ GRA_Lanczos = 4
	
	#note the size of the cell is user dependent
	#since a MODIS grid should represent a 10 x 10 deg grid
	#cell_size = 10 deg/2400 number of cells = 0.00416666 deg/cell
	
	#outputs the MODIS data as WGS84
	'''
	geo_t = mod_data.GetGeoTransform()
	#wkt = sds.GetProjection()
	#use the projection from the http://daac.ornl.gov/MODIS/modis.prj webpage
	#prj ='PROJCS["Sinusoidal",GEOGCS["GCS_Undefined",DATUM["Undefined",SPHEROID["User_Defined_Spheroid",6371007.181,0.0]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Sinusoidal"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],UNIT["Meter",1.0]]'
	wgs84 = osr.SpatialReference()
	#wgs84.ImportFromEPSG(4326) #doesn't work for some reason
	wgs84.SetWellKnownGeogCS('WGS84')
	#wgs84.ExportToPrettyWkt() #displays the well known transform to check
	proj = mod_data.GetProjectionRef()
	src = osr.SpatialReference()
	src.ImportFromWkt(proj)
	#src.ExportToPrettyWkt()
	tx = osr.CoordinateTransformation(src,wgs84)
	x_size = mod_data.RasterXSize # Raster xsize
	y_size = mod_data.RasterYSize # Raster ysize
	# Work out the boundaries of the new dataset in the target projection
	(ulx, uly, ulz) = tx.TransformPoint(geo_t[0], geo_t[3])
	(lrx, lry, lrz) = tx.TransformPoint(geo_t[0] + geo_t[1]*x_size, geo_t[3] + geo_t[5]*y_size)
	#create in-memory raster
	mem_drv = gdal.GetDriverByName('MEM')
	pixel_spacing = cell_size 
	dest = mem_drv.Create('', int((lrx - ulx)/pixel_spacing), int((uly - lry)/pixel_spacing), 1, gdal.GDT_Float32)
	# Calculate the new geotransform
	new_geo = (ulx, pixel_spacing, geo_t[2], uly, geo_t[4], -pixel_spacing)
	# Set the geotransform
	dest.SetGeoTransform(new_geo)
	dest.SetProjection(src.ExportToWkt())
	res = gdal.ReprojectImage(mod_data, dest, src.ExportToWkt(), wgs84.ExportToWkt(), method)
	return(dest)

def reproj_array(mod_array, nbands = 1, cell_size = 0.004166666666666667, method =  0):
	out_array = []
	for i in range(nbands):
		reproj_ds = reproj_wgs84(mod_array[i], cell_size = cell_size, method = method)
		out_array.append(reproj_ds)
	return(out_array)

def mosaic(ds1, ds2,stats_out = False):
	#takes in 2 gdal datasets to mosaic together
	band1 = ds1.GetRasterBand(1)
	rows1 = ds1.RasterYSize
	cols1 = ds1.RasterXSize
	# get the corner coordinates for doq1
	transform1 = ds1.GetGeoTransform()
	minX1 = transform1[0]
	maxY1 = transform1[3]
	pixelWidth1 = transform1[1]
	pixelHeight1 = transform1[5]
	maxX1 = minX1 + (cols1 * pixelWidth1)
	minY1 = maxY1 + (rows1 * pixelHeight1)
	# read in doq2 and get info about it
	band2 = ds2.GetRasterBand(1)
	rows2 = ds2.RasterYSize
	cols2 = ds2.RasterXSize
	# get the corner coordinates for doq2
	transform2 = ds2.GetGeoTransform()
	minX2 = transform2[0]
	maxY2 = transform2[3]
	pixelWidth2 = transform2[1]
	pixelHeight2 = transform2[5]
	maxX2 = minX2 + (cols2 * pixelWidth2)
	minY2 = maxY2 + (rows2 * pixelHeight2)
	# get the corner coordinates for the output
	minX = min(minX1, minX2)
	maxX = max(maxX1, maxX2)
	minY = min(minY1, minY2)
	maxY = max(maxY1, maxY2)
	# get the number of rows and columns for the output
	cols = int(np.ceil((maxX - minX) / pixelWidth1))
	rows = int(np.ceil(((maxY - minY) / abs(pixelHeight1))))
	# compute the origin (upper left) offset for doq1
	xOffset1 = int((minX1 - minX) / pixelWidth1)
	yOffset1 = int((maxY1 - maxY) / pixelHeight1)
	# compute the origin (upper left) offset for doq2
	xOffset2 = int((minX2 - minX) / pixelWidth1)
	yOffset2 = int((maxY2 - maxY) / pixelHeight1)
	# create the output image
	mem_drv = gdal.GetDriverByName('MEM')
	dsOut = mem_drv.Create('', cols, rows, 1, gdal.GDT_Float32)
	bandOut = dsOut.GetRasterBand(1) #empty for now
	# read in doq1 and write it to the output
	data1 = band1.ReadAsArray(0, 0, cols1, rows1)
	bandOut.WriteArray(data1, xOffset1, yOffset1)
	# read in doq2 and write it to the output
	data2 = band2.ReadAsArray(0, 0, cols2, rows2)
	bandOut.WriteArray(data2, xOffset2, yOffset2)
	# compute statistics for the output
	bandOut.FlushCache()
	stats = bandOut.GetStatistics(0, 1)
	# set the geotransform and projection on the output
	geotransform = [minX, pixelWidth1, 0, maxY, 0, pixelHeight1]
	dsOut.SetGeoTransform(geotransform)
	dsOut.SetProjection(ds1.GetProjection())
	if stats_out:
		return(dsOut, stats)
	else:
		return(dsOut)

def create_band_list(first_band = 1, last_band = 7, qc= True):
	'''
	defaul returned 
	band_list = [1,2,3,4,5,6,7,'qc']
	'''
	band_list = list(range(first_band,last_band+1))
	if qc:
		band_list.append('qc') #add the quality control band
	return(band_list)

def mosaic_files(files_to_mosaic, first_band =1, last_band = 7, qc = True, reproj = False, method = 0):
	'''
	input a list of files to mosaic together
	returns the mosaiced files and bands
	'''
	n_files = len(files_to_mosaic) #there are always 2 in this case
	mod_bands = [[] for i in range(n_files)]
	#make a band list
	band_list = create_band_list(first_band = first_band, last_band = last_band, qc = qc)
	for j in range(n_files):
		for k in band_list:
			mod_bands[j].append(moda.mod_acquire_by_band(files_to_mosaic[j], k))
	processed_mosaics = []
	for band in range(len(mod_bands[0])):
		ds = []
		for tile in range(n_files):
			ds.append(mod_bands[tile][band])
		mod_mosaic = mosaic(ds[0], ds[1])
		#now transform the mosaics
		if reproj:
			processed_mosaics.append(reproj_wgs84(mod_mosaic, method = method))
		else:
			processed_mosaics.append(mod_mosaic)
	return(processed_mosaics)

def clip_wgs84_scene(aoa, mod_data):
	'''
	input area of analysis to be clipped to in WGS84 coordinates
	aoa = [xmin, xmax, ymin, ymax]
	and scene that has already been projected to WGS84
	outputs the clipped scene as a gdal Dataset
	'''
	grid = mod_data.ReadAsArray()
	geo_t, proj, x_size, y_size, dtype = moda.get_geo_info(mod_data)
	cell_size = geo_t[1]
	xul = geo_t[0] 
	yul = geo_t[3]
	xlr = xul + (cell_size * x_size)
	ylr = yul - (cell_size * y_size)
	xmin_i = (aoa[0] - xul)/cell_size
	xmax_i = (aoa[1] - xul)/cell_size
	ymin_i = (yul - aoa[2] )/cell_size
	ymax_i = (yul - aoa[3])/cell_size
	i_s = np.array([np.floor(xmin_i), np.ceil(xmax_i), np.ceil(ymin_i), np.floor(ymax_i)]).astype(int)
	clip = grid[i_s[3]:i_s[2], i_s[0]:i_s[1]]
	
	lat_array = np.arange(x_size)*geo_t[1] + geo_t[0]
	lon_array = np.arange(y_size)*geo_t[5] + geo_t[3]
	clip_xul = lat_array[i_s[0]]
	clip_yul = lon_array[i_s[3]]
	clip_y_size = i_s[2] -i_s[3]
	clip_x_size = i_s[1] -i_s[0]
	clip_geo_t = (clip_xul, cell_size, 0.0, clip_yul, 0.0, -1.0*cell_size)
	
	mem_drv = gdal.GetDriverByName('MEM')
	dest = mem_drv.Create('', int(clip_x_size), int(clip_y_size), 1, dtype)
	dest.SetGeoTransform(clip_geo_t)
	dest.SetProjection(proj.ExportToWkt())
	dest.GetRasterBand(1).WriteArray(clip)
	#geotransform is in form (xul, xcell_size, xoffset, yul, yoffset, ycell_size)
	return(dest)

def disturbance_index(gr, br, we):
	'''
	input the Tasseled cap transformation of greenness, brightness, and wetness 
	and returns the Disturbance Index as defined by Healey et al 2005 for Landsat
	'''
	return(br - (gr + we))
