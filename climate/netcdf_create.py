#create some netCDF4 files to merge the data together....
from netCDF4 import Dataset
import numpy as np


if __name__ == '__main__':
	startyear = 1948
	endyear = 2012
	referencefile = 'E:\\TOPOWX\\annual\\tmin\\tmin_%s.nc' %(str(startyear))
	r_ds = Dataset(referencefile)
	lat_len = len(r_ds.variables['lat'])
	lon_len = len(r_ds.variables['lon'])
	time_len = len(r_ds.variables['time']) #note that the days account for leap year...
	ncfile = Dataset('temp.nc','w')
	print('creating file')
	lat_dim = ncfile.createDimension('lat', lat_len)     # latitude axis
	lon_dim = ncfile.createDimension('lon', lon_len)     # longitude axis
	time_dim = ncfile.createDimension('time', 0)   # unlimited axis if it is assigned as None or 0
	print('-- Created dimensions')
	# Define two variables with the same names as dimensions,
	# a conventional way to define "coordinate variables".
	lat = ncfile.createVariable('lat', 'f4', ('lat',))
	lat.units = 'degrees_north'
	lat.standard_name = 'latitude'
	lon = ncfile.createVariable('lon', 'f4', ('lon',))
	lon.units = 'degrees_east'
	lon.standard_name = 'longitude'
	time = ncfile.createVariable('time', 'f4', ('time',))
	time.units = 'days since %s-1-1 0:0:0' %(str(startyear))
	time.standard_name = 'time'
	tmin = ncfile.createVariable('tmin','f4',('time','lat','lon'))
	tmin.units = 'C' #Celcius
	tmin.standard_name = 'air_temperature'
	print('-- Created variables with attributes')
	#======================================
	# Write latitudes, longitudes.
	# Note: the ":" is necessary in these "write" statements
	lat[:] = r_ds.variables['lat'][:]
	lon[:] = r_ds.variables['lon'][:]
	#======================================
	# Write data
	ntimes = time_len
	nlats = lat_len
	nlons = lon_len
	#if building an empty array just use zeros for the shape of (time, lats, lons)
	tmin[:,:,:] = r_ds.variables['tmin']
	print("-- Wrote data, tmin.shape is now ", tmin.shape)
	ncfile.close()
	print("-- File closed successfully")