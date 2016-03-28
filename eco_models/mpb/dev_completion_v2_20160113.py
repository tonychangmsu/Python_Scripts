#Title: 	dev_completion_v2.py
#Author:	Tony Chang
#Date: 		2015.11.10
#Abstract:	library to determine when the development will complete for an adult MPB 

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import scipy.integrate as sp
import timeit
from numba import double, jit
import gdal as gdal

def dev_trapz(ds):
	d = ds.variables['development'][:]/1e4
	t = ds.variables['time'][:]
	t_floor = np.floor(t).astype(int)
	#now set up a loop to solve the index for each whole integer or iterate up
	#we can figure out if we have a leap year or not also by looking at the max day
	#i.e. np.max(t_floor) == 365 , means this is a leap year dataset

	max_days = np.max(t_floor) + 1
	z,m,n = d.shape
	daily_dev = np.zeros((max_days, m,n))
	start = 0 #first integration point
	k_array = np.zeros(max_days)
	for k in range(max_days):
		ones = np.ones((m,n))
		if k == max_days-1:
			tm = np.tensordot(t[start:], ones, axes=0)
			daily_dev[k] = np.trapz(d[start:],tm,axis=0) 
		else:
			end = np.argmax(t_floor>k)
			tm = np.tensordot(t[start:end+1], ones, axes=0) #plus one to integrate the end point
			daily_dev[k] = np.trapz(d[start:end+1],tm,axis=0) 
			#define the new start
			start = end
		k_array[k] = k+1 #counter for each day
	return(daily_dev, k_array)

def write_daily_dev_data(nc_ds, daily_dev, k_array, outname):
	#inputs the topowx hourly data as a reference for the development rate write
	#outputs the development rate into a netCDF4 file on disk
	year = int(nc_ds.variables['time'].units[11:15])
	lat_array = nc_ds.variables['latitude']
	lon_array = nc_ds.variables['longitude']
	
	root_grp = nc.Dataset(outname, 'w', format=	'NETCDF4')
	root_grp.description = 'Cumulative development per day'
	root_grp.history = 'Created %s' %(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
	root_grp.source = 'Montana State University Landscape Biodiversity Lab'

	# dimensions
	root_grp.createDimension('time', None)
	root_grp.createDimension('lat', len(nc_ds.variables['latitude'][:]))
	root_grp.createDimension('lon', len(nc_ds.variables['longitude'][:]))

	# variables
	times = root_grp.createVariable('time', 'f4', ('time',))
	latitudes = root_grp.createVariable('latitude', 'f4', ('lat',))
	longitudes = root_grp.createVariable('longitude', 'f4', ('lon',))
	daily_development = root_grp.createVariable('development', 'f4', ('time', 'lat', 'lon',))
	
	# descriptions
	latitudes.units = 'degrees_north'
	longitudes.units = 'degrees_east'
	daily_development.units = 'cumulative development per day'
	times.units = 'Days since %s-1-1 0:0:0'%(year)
	times.calendar = 'standard'
	
	latitudes[:] = nc_ds.variables['latitude'][:]
	longitudes[:] = nc_ds.variables['longitude'][:]
	times[:] = k_array
	daily_development[:,:,:] = daily_dev
	root_grp.close()
	return(print("%s written!"%(outname)))
	
######################### MAIN ########################
#first get the data
for stage in range(1,9):
	for year in range(1948, 1980):
		filename = "K:\\NASA_data\\mpb_phen_out\\hourly\\stage_%s\\mpb_phen_stage_%s_%s.nc"%(stage,stage,year)
		ds = nc.Dataset(filename)
		daily_dev, k_array = dev_trapz(ds)
		#outname =  "K:\\NASA_data\\mpb_phen_out\\daily\\stage_%s\\mpb_phen_stage_%s_%s.nc"%(stage,stage,year)
		outname =  "G:\\MPB\\daily\\stage_%s\\mpb_phen_stage_%s_%s.nc"%(stage,stage,year)
		write_daily_dev_data(ds, daily_dev, k_array, outname) 
		#output file is only 338,512 KB, totalling ~92 GB!
