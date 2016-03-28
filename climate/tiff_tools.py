# Title: GeoTiff directory edit function 
# Author: Tony Chang
# Date: 07.28.2013
# Abstract: Short function to search through a directory, extract the tiffs and manipulate them all in the same
# All tiffs must be of same dimensions. 

import glob 
import os
import numpy as np
import gdal 
from gdalconst import *
import osr 

def tiffreader(pathname): 
#input the full path name of the file as a string!
	ds = gdal.Open(pathname)
	tif = np.array(ds.GetRasterBand(1).ReadAsArray())
	return(tif)
	
def dirlist(dirname):
#gets all the tif files within the search directory
	l = os.listdir(dirname)
	tiflist = []
	for i in l:
		if i[-3:] == 'tif':
			tiflist.append(i)
	return(tiflist)

def tiffwrite(data, head, path, name = 'Temp'):
	xll = head['xul']
	yul = head['yul']
	cellsize = head['csize']
	nbands = head['bands']
	Nx = head['ncols']
	Ny = head['nrows']
	fileformat = "GTiff"
	driver = gdal.GetDriverByName(fileformat)
	geotransform = [xll, cellsize,0.0,yul, 0.0, -cellsize]
	srs = osr.SpatialReference()
	writename = path + name 
	outDs = driver.Create(writename, Nx, Ny, nbands, gdal.GDT_Float32)
	outDs.SetGeoTransform(geotransform)
	srs.SetWellKnownGeogCS("WGS72")
	outDs.SetProjection(srs.ExportToWkt())
	for band in range(nbands):
		outBand = outDs.GetRasterBand(band+1)
		outBand.SetNoDataValue(-9999)
		outBand.WriteArray(data,0,0)
	outDs = None
	return(print(writename + " filebuilt! \n"))

def genvarlist(varname, bm, em):
	l = []
	for i in range(bm, em+1):
		l.append(varname + str(i) + ".tif")
	return(l)

def averagetif(pathname, tiflist):
	temp = tiffreader(pathname+ tiflist[0])
	sum = np.zeros(np.shape(temp))
	for i in tiflist:
		sum += tiffreader(pathname+ i)
	return(sum/len(tiflist))
		
		
#------------------main----------------
h = Headerextract()
p = 'd:\\chang\\gis_data\\se_data\\climate\\'
tl = dirlist(p)

'''
for i in tl:
    temp = tiffreader(p+i)
    temp[ele == 32767] = np.nan
	tiffwrite(temp,h, p+"07282013\\", i)
'''    