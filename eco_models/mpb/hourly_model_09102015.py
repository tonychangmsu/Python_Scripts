#Title: hourly_model.py
#Author: Tony Chang
#Date: 9/2/2015
#Abstract: Test code to transform daily temperature values into hourly under various algorithms

#Newton's Law of Cooling method
#dP/dt = k(P-A)
#where A is the ambient temperature, P is the phloem temperature, k is the rate of temperature transfer from tree to phloem
#from regression estimates, k = 0.5258 and k = 1.3357 under the Vienna and Ranch parameterizations respectively
#Newton's models
#P(t + del_t) = P(t) + k[P(t) -A(t)] del_t
#take del_t = 1 since minimum time step is 1 hour
#this will track Northern aspects well, but requires hourly temperatures (which are unavailable)
#
#alternatively we can use the Cosine method
#
#P_t = 0.5*(P_max + P_min) + 0.5*(P_max - P_min) * cos(pi + (pi*(t-t_dmin)/(t_dmax-t_dmin)) for t in (t_dmin, t_dmax)
# where P_max is the max daily temp, P_min is the min daily temp, 
# t is the julian hour, t_dmin is the julian hour for day when the minimum temperature is reached,
# t_dmax is the julian hour for day when the maximum temperatuer is reached

import numpy as np
import matplotlib.pyplot as plt
import timeit 
import os
import netCDF4 as nc
os.chdir('E:\\TOPOWX')
import geotool as gt
import util_dates as utld
import gdal as gdal
import time
import pandas as pd
import datetime as datetime
import pytz as pytz
import calendar as calendar

def cos_transform(P_max, P_min, t_dmin, t_dmax, t_dmin_next, t):
	#function for transforming daily data to hourly using a cosine function
	#if (t<t_dmin):
	#	P_t = 0.5*(P_max + P_min) + 0.5*(P_max - P_min) * cos(((pi*(t-t_dmax))/((t_dmin-1)-t_dmax))) #not right yet....
	#if ((t>t_dmin_prev) and (t<t_dmax)):
	if (t<t_dmax):
		P_t = 0.5*(P_max + P_min) + 0.5*(P_max - P_min) * cos(pi + ((pi*(t-t_dmin))/(t_dmax-t_dmin)))
	elif (t>t_dmax):
		P_t = 0.5*(P_max + P_min) + 0.5*(P_max - P_min) * cos(((pi*(t-t_dmax))/((t_dmin_next)-t_dmax)))
	return(P_t)	

def cos_transform_array(P_max, P_min, t_dmin, t_dmax, t_dmin_next, t):
	#perform for an array
	out = np.zeros(np.shape(P_max))
	out[t<t_dmax] = 0.5*(P_max[t<t_dmax] + P_min[t<t_dmax]) + 0.5*(P_max[t<t_dmax] - P_min[t<t_dmax]) * np.cos(np.pi + ((np.pi*(t[t<t_dmax]-t_dmin[t<t_dmax]))/(t_dmax[t<t_dmax]-t_dmin[t<t_dmax])))
	out[t>t_dmax] = 0.5*(P_max[t>t_dmax] + P_min[t>t_dmax]) + 0.5*(P_max[t>t_dmax] - P_min[t>t_dmax]) * np.cos(((np.pi*(t[t>t_dmax]-t_dmax[t>t_dmax]))/((t_dmin_next[t>t_dmax])-t_dmax[t>t_dmax])))
	return(out)
	
	
def ds_unix_time_convert(ds, ds_variable, julian_date, year, tz = "US/Mountain"):
	#takes the netcdf dataset and converts it to hours for the given julien date
	HOURS_PER_SECOND = 1/(60*60)
	date_info = date_info = '%s%s'%(ds.variables['time'].units[11:15], ds.variables['time'][julian_date-1]+1)
	j_date = datetime.datetime.strptime(date_info, '%Y%j')
	local_tz = pytz.timezone(tz)
	local_dt = local_tz.localize(j_date) #assign the time zone
	utc_dt = local_dt.astimezone(pytz.utc) 
	unixtime_local = calendar.timegm(local_dt.timetuple())
	unixtime_utc = calendar.timegm(utc_dt.timetuple()) 
	#month and date convert to unixtime at 00:00:00
	#subtract this unixtime_utc from the data arrays
	#divide data by 3600 to get hours
	out_ds = (ds.variables[ds_variable][julian_date-1] - unixtime_utc)* HOURS_PER_SECOND
	return(out_ds)

def hourly_transform(year, AOA, t_dmin_prev = None, npartitions = 24):
	workspace = 'E:\\TOPOWX\\annual\\'
	
	nc_tmax_name = '%s%s\\%s_%s.nc' %(workspace, 'tmax', 'tmax', year) #use tmax
	nc_tmin_name = '%s%s\\%s_%s.nc' %(workspace, 'tmin', 'tmin', year)
	nc_ds = nc.Dataset(nc_tmax_name)
	max_x_i = np.where(nc_ds.variables['lon'][:]>=AOA[2])[0][0]
	min_x_i = np.where(nc_ds.variables['lon'][:]>=AOA[0])[0][0]
	min_y_i = np.where(nc_ds.variables['lat'][:]<=AOA[3])[0][0]
	max_y_i = np.where(nc_ds.variables['lat'][:]<=AOA[1])[0][0]
	
	ndays = len(nc_ds.variables['time'])
	dim = np.shape(nc_ds.variables['tmax'][0,min_y_i:max_y_i,min_x_i:max_x_i])
	
	P_max = nc_ds.variables['tmax'][:,min_y_i:max_y_i,min_x_i:max_x_i]
	P_min = nc.Dataset(nc_tmin_name).variables['tmin'][:,min_y_i:max_y_i,min_x_i:max_x_i]
	
	del nc_ds #clear memory once we get the temperature data
	julian_day_start = 1
	julian_day_end = ndays #this should be determined by the length of the climate data stack
	
	nhours = (julian_day_end-julian_day_start) * npartitions
	hourly_temp = np.zeros((nhours,dim[0],dim[1])).astype(int) #storage array for our output (as int type for memory saving)
	hours = np.zeros((nhours,dim[0],dim[1])).astype(int) #hours are initialized as 0
	unit_hours = np.zeros((nhours)) #unit hours are initialized as 0
	
	sp_filename = 'K:\\NASA_data\\solar_out\\%s_snsrss.nc'%(year)
	sp_ds = nc.Dataset(sp_filename)
	#need to find the high and low points and divide into 24 periods?
	if t_dmin_prev == None:
		t_dmin_prev = np.ones(dim) * (julian_day_start-1) #initialize the first minimum if there is none entered
	for i in range(julian_day_start, julian_day_end):
		#get the sun position arrays
		##what if we just kept time units in hours?
		sunrise = ds_unix_time_convert(sp_ds, 'sunrise', i-1, year) /24 #in days
		transit = ds_unix_time_convert(sp_ds, 'transit', i-1, year) /24
		sunset = ds_unix_time_convert(sp_ds, 'sunset', i-1, year) /24
		if (i == julian_day_end-1): #get sunrise from the next year
			sp_next = 'K:\\NASA_data\\solar_out\\%s_snsrss.nc'%(year+1)
			sp_ds_next = nc.Dataset(sp_next)
			sunrise_next = ds_unix_time_convert(sp_ds_next, 'sunrise', 0, year+1) /24
		else:
			sunrise_next = ds_unix_time_convert(sp_ds, 'sunrise', i, year) /24
		day_hour = i-1 #current calculated day in hours
		t_dmax = (sunset - transit)/2 + transit + day_hour #calculate the time when maximum temperature is reached in hours
		t_dmin = sunrise + day_hour #calculate the time when minimum temperature is reached
		t_dmin_next = sunrise_next + day_hour + 1 #add one for the next day
		#so we can make a time array that starts at the t_dmin_prev and then goes to the last t_dmin_next. 
		#but we would like to start there and then go to each hourly increment (1/24) of a day...
		#this is not the most easy to perform maybe?
		#so get the decimal position, then count up to mext?
		bound_time = np.tensordot(np.linspace(0,1,npartitions+1),(t_dmin_next-t_dmin_prev),axes=0)  + t_dmin_prev 
		##is there some ability to get to the rounded hour for each of there, the problem is that all the times are
		##as a fraction of the day....or another option is just to round that hour time to the nearest hour as a 
		##representation. That may be a good way to go about it...
		##it seems that the first 2 days are a bit strange, but then the hours work out after a few iterations...
		##next step is to run for 2 years continuously and check if all the hours make sense...
		##then I will begin saving the data.
		##basically when julian day is equal to last day, then get the sunrise next data and temperature date from
		##the next file (accounting for leap year)s
		#first off, maybe change all the values that are not the first nor last data point into whole integer values
		#that way we can reference them?
		
		for t_hour in range(len(bound_time)): #twenty four hour day division
			index = day_hour * npartitions + t_hour 
			#print(index)
			#current_julian_hour = day_hour + t_hour/24 * np.ones(dim)
			current_julian_hour = bound_time[t_hour]
			current_P = cos_transform_array(P_max[i-1], P_min[i-1], t_dmin, t_dmax, t_dmin_next, current_julian_hour)
			if ((i != julian_day_start) and (t_hour == 0)): #for the first new day
				hourly_temp[index] = ((current_P + last_P)/2) * 100 # take the average between two days, multiply by 1000 for int
				hours[index] = current_julian_hour * 100
				#unit_hours[index] = np.mean((current_julian_hour-np.floor(current_julian_hour))*24, axis = (1,2)) #return the hourly time in 24 format
				unit_hours[index] = np.mean(current_julian_hour) #return the hourly time in 24 format
			elif (t_hour == len(bound_time)-1): #last hour
				last_P = current_P 
				#in the last hour case, we only save the last_P for calculation and 
				#apply it for the next day iteration, so bound_time is just a little bit bigger.
			else:
				hourly_temp[index] = current_P * 100
				hours[index] = current_julian_hour * 100
				#unit_hours[index] = np.mean((current_julian_hour-np.floor(current_julian_hour))*24, axis = (1,2)) #return the hourly time in 24 format
				unit_hours[index] = np.mean(current_julian_hour) #return the hourly time in 24 format
		t_dmin_prev = bound_time[-1]
	#not bad, 302.321 secs for a whole year of calculations. 
	#h_time = hours.mean(axis =(1,2))
	#t_hourly = hourly_temp.mean(axis=(1,2))

	##still not working right....@t.chang 09.02.2015// working now 09.08.2015
	##looks half decent!
	##so now we have an option of saving these hourly temperature rasters for the GYE. 
	##this may not be so bad since again our domain size if much smaller than the entire country...
	##the problem is to unify the time periods to a single time, and not have the spread out time raster...
	##secondly, we need to burn just the first day of the 1948 raster because the minimum temperature time is off?
	return(hourly_temp, hours,  unit_hours, t_dmin_prev)
	
def write_data(outname, h_temp, unit_hours):
	#now we need to adapt this to consider more days
	#open the topowx data
	workspace = 'E:\\TOPOWX\\annual\\'
	csize = 0.00833333333
	xmax = -108.19583334006; xmin = -112.39583333838; ymin = 42.270833326049996; ymax = 46.19583332448 # GYE bounds
	AOA = [xmin, ymin, xmax, ymax] #specify the bounds for the FIA data
	filename = '%s%s\\%s_%s.nc' %(workspace, 'tmin', 'tmin', 1948)
	i = 0
	nc_ds = nc.Dataset(filename)
	max_x_i = np.where(nc_ds.variables['lon'][:]>=AOA[2])[0][0]
	min_x_i = np.where(nc_ds.variables['lon'][:]>=AOA[0])[0][0]
	min_y_i = np.where(nc_ds.variables['lat'][:]<=AOA[3])[0][0]
	max_y_i = np.where(nc_ds.variables['lat'][:]<=AOA[1])[0][0]

	#ideally, the transform function should work for everyday of the topowx file
	#okay lets try for january 1st and june 30
	lat_array = nc_ds.variables['lat'][min_y_i:max_y_i]
	lon_array = nc_ds.variables['lon'][min_x_i:max_x_i]
	lon, lat = np.meshgrid(lon_array, lat_array)
	grid_dimension = (len(lat_array), len(lon_array))
	root_grp = nc.Dataset(outname, 'w', format='NETCDF4')
	root_grp.description = 'Temperature'
	root_grp.history = 'Created %s' %(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
	root_grp.source = 'Montana State University Landscape Biodiversity Lab'

	# dimensions
	root_grp.createDimension('time', None)
	root_grp.createDimension('lat', grid_dimension[0])
	root_grp.createDimension('lon', grid_dimension[1])

	# variables
	times = root_grp.createVariable('time', 'f4', ('time',))
	latitudes = root_grp.createVariable('latitude', 'f4', ('lat',))
	longitudes = root_grp.createVariable('longitude', 'f4', ('lon',))
	temperature = root_grp.createVariable('temperature', 'i', ('time', 'lat', 'lon',))
	days = root_grp.createVariable('days', 'f4', ('time', 'lat', 'lon',)) #may not want this variable...
	
	# descriptions
	latitudes.units = 'degrees_north'
	longitudes.units = 'degrees_east'
	temperature.units = 'deg C * 100'
	days.units = 'Days since %s-1-1 0:0:0'%(year)
	times.units = 'Grid of days since %s-1-1 0:0:0'%(year)
	times.calendar = 'standard'
	temperature = h_temp
	times = unit_hours
	root_grp.close()	
	
if __name__ == "__main__":

	xmax = -108.19583334006; xmin = -112.39583333838; ymin = 42.270833326049996; ymax = 46.19583332448 # GYE bounds
	AOA = [xmin, ymin, xmax, ymax] #specify the bounds for the FIA data
	year = 1948
	tin = timeit.time.time()
	h_temp, hours, unit_hours, t_dmin_prev = hourly_transform(year, AOA)
	tout=timeit.time.time()
	print('Time = %s' %(tout-tin))
	#about 9.3 mins to perform a single year
	#now write a netcdf4 file of it
	
	#looks good!
	#so package this into a function

	writespace = 'K:\\NASA_data\\hourly_topowx\\'
	writefile = '%s_topowx_hourly_temp.nc'%(year)
	outname = '%s%s'%(writespace, writefile)
	write_data(outname, h_temp, unit_hours)
			