#Title: grid_write.py
#Author: Tony Chang
#Abstract: Functions saving gridded data files in multiple formats
#Created Date: 02/01/2016

#local directory : K:\\NASA_data\\scripts
#%load_ext autoreload
#%autoreload 2

from osgeo import gdal, osr
import numpy as np
import MODIS_acquire as moda
import netCDF4 as nc
import datetime

def tiff_write(data,Nx,Ny,cellsize,yul,xll, writename):   
	fileformat = "GTiff"
	nbands = 1
	driver = gdal.GetDriverByName(fileformat)
	geotransform = [xll, cellsize,0.0,yul, 0.0, -cellsize]
	srs = osr.SpatialReference()
	outDs = driver.Create(writename, Nx, Ny, nbands, gdal.GDT_Float32)
	outDs.SetGeoTransform(geotransform)
	srs.SetWellKnownGeogCS("WGS84")
	outDs.SetProjection(srs.ExportToWkt())
	for band in range(nbands):
		outBand = outDs.GetRasterBand(band+1)
		outBand.SetNoDataValue(-9999)
		outBand.WriteArray(data,0,0)
	outDs = None
	return(print(writename + " filebuilt!\n"))

def tiff_write_gdal(data, writename):
	geo_t, proj, x_size, y_size, dtype = moda.get_geo_info(data)
	fileformat = "GTiff"
	nbands = 1
	driver = gdal.GetDriverByName(fileformat)
	outDs = driver.Create(writename, x_size, y_size, nbands, dtype)
	outDs.SetGeoTransform(geo_t)
	outDs.SetProjection(proj.ExportToWkt())
	for band in range(nbands):
		outBand = outDs.GetRasterBand(band+1)
		outBand.SetNoDataValue(-9999)
		outBand.WriteArray(data.ReadAsArray(),0,0)
	outDs = None
	return(print(writename + " filebuilt!\n"))

def netcdf4_write(value_data, year, julian_date, outname, appnd = False):
	if appnd: 
		root_grp = nc.Dataset(outname, 'a')
		#find length and add the data
		n = len(root_grp.variables['date'])
		root_grp.variables['date'][n] = julian_date
		root_grp.variables['brightness'][n,:,:] = value_data[0].ReadAsArray()
		root_grp.variables['greenness'][n,:,:] = value_data[1].ReadAsArray()
		root_grp.variables['wetness'][n,:,:] = value_data[2].ReadAsArray()
		root_grp.variables['qc'][n,:,:] = value_data[3].ReadAsArray()
		root_grp.close()
		return(print('%s updated!' %(outname)))

	else: #write a new file
		geo_t, proj, x_size, y_size, dtype = moda.get_geo_info(value_data[0])
		lon_array = np.arange(x_size)*geo_t[1] + geo_t[0]
		lat_array = np.arange(y_size)*geo_t[5] + geo_t[3]
		root_grp = nc.Dataset(outname, 'w', format = 'NETCDF4')
		root_grp.description = 'Tasselled Cap Transform values' 
		root_grp.history = 'Created %s' %(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
		root_grp.source = 'Montana State University Landscape Biodiversity Lab'

		# dimensions
		root_grp.createDimension('date', None) #infinite dimension
		root_grp.createDimension('lat', len(lat_array))
		root_grp.createDimension('lon', len(lon_array))

		# variables
		date = root_grp.createVariable('date', 'uint16', ('date',))
		latitudes = root_grp.createVariable('latitude', 'f4', ('lat',))
		longitudes = root_grp.createVariable('longitude', 'f4', ('lon',))
		br = root_grp.createVariable('brightness', 'f4', ('date', 'lat', 'lon',))
		gr = root_grp.createVariable('greenness', 'f4', ('date', 'lat', 'lon',))
		we = root_grp.createVariable('wetness', 'f4', ('date', 'lat', 'lon',))
		qc = root_grp.createVariable('qc', 'uint32', ('date', 'lat', 'lon',))
		
		# descriptions
		latitudes.units = 'degrees_north'
		longitudes.units = 'degrees_east'
		br.units = 'Tasselled Cap Transform brightness index'
		gr.units = 'Tasselled Cap Transform greenness index'
		we.units = 'Tasselled Cap Transform wetness index'
		qc.units = 'MOD09A1 quality control value'
		date.units = 'Julian days since %s 0:0:0'%(year)
		date.calendar = 'Julian calendar'
		
		latitudes[:] = lat_array[:]
		longitudes[:] = lon_array[:]
		date[0] = julian_date #initial date
		br[0,:,:] = value_data[0].ReadAsArray() #initial brightness
		gr[0,:,:] = value_data[1].ReadAsArray() #initial greenness
		we[0,:,:] = value_data[2].ReadAsArray() #initial development_date
		qc[0,:,:] = value_data[3].ReadAsArray() #initial development_date
		root_grp.close()
		return(print("%s written!"%(outname)))