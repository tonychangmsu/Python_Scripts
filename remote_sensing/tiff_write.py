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

def netcdf_write(data, date, outname, appnd = False):
	if appnd: 
		root_grp = nc.Dataset(outname, 'a')
		#find length and add the data
		n = len(root_grp.variables['date'])
		root_grp.variables['date'][n] = date
		root_grp.variables['value'][n,:,:] = data 
		root_grp.close()
		return(print('%s updated!' %(outname)))

	else: #write a new file
		lat_array = nc_ds.variables['latitude']
		lon_array = nc_ds.variables['longitude']
		
		root_grp = nc.Dataset(outname, 'w', format = 'NETCDF4')
		root_grp.description = 'Tasselled Cap Transform values' 
		root_grp.history = 'Created %s' %(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
		root_grp.source = 'Montana State University Landscape Biodiversity Lab'

		# dimensions
		root_grp.createDimension('time', None) #infinite dimension
		root_grp.createDimension('lat', len(nc_ds.variables['latitude'][:]))
		root_grp.createDimension('lon', len(nc_ds.variables['longitude'][:]))

		# variables
		times = root_grp.createVariable('time', 'f4', ('time',))
		latitudes = root_grp.createVariable('latitude', 'f4', ('lat',))
		longitudes = root_grp.createVariable('longitude', 'f4', ('lon',))
		value = root_grp.createVariable('value', 'uint16', ('time', 'lat', 'lon',)) #unsigned 16 int
		
		# descriptions
		latitudes.units = 'degrees_north'
		longitudes.units = 'degrees_east'
		value.units = 'tasselled cap transform index'
		times.units = 'Julian days since %s-%s 0:0:0'%(year, julian_start)
		times.calendar = 'Julian calendar'
		
		latitudes[:] = nc_ds.variables['latitude'][:]
		longitudes[:] = nc_ds.variables['longitude'][:]
		times[0] = julian_start #initial time
		development_date[0,:,:] = dev_comp_date #initial development_date
		root_grp.close()
		return(print("%s written!"%(outname)))