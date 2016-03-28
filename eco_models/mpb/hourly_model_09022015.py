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

def main():
	#now we need to adapt this to consider more days
	#open the topowx data
	workspace = 'E:\\TOPOWX\\annual\\'
	year = 1948
	
	filename = '%s%s\\%s_%s.nc' %(workspace, 'tmin', 'tmin', year)
	xmax = -108.19583334006; xmin = -112.39583333838; ymin = 42.270833326049996; ymax = 46.19583332448 # GYE bounds
	AOA = [xmin, ymin, xmax, ymax] #specify the bounds for the FIA data
	nc_ds = nc.Dataset(filename)
	max_x_i = np.where(nc_ds.variables['lon'][:]>=AOA[2])[0][0]
	min_x_i = np.where(nc_ds.variables['lon'][:]>=AOA[0])[0][0]
	min_y_i = np.where(nc_ds.variables['lat'][:]<=AOA[3])[0][0]
	max_y_i = np.where(nc_ds.variables['lat'][:]<=AOA[1])[0][0]
	
	ndays = len(nc_ds.variables['time'])
	dim = np.shape(nc_ds.variables['tmin'][0,min_y_i:max_y_i,min_x_i:max_x_i])
	del nc_ds #remove this data file which is being used for reference
	
	tmax_name = '%s%s\\%s_%s.nc' %(workspace, 'tmax', 'tmax', year)
	P_max = nc.Dataset(tmax_name).variables['tmax'][:,min_y_i:max_y_i,min_x_i:max_x_i]

	tmin_name = '%s%s\\%s_%s.nc' %(workspace, 'tmin', 'tmin', year)
	P_min = nc.Dataset(tmin_name).variables['tmin'][:,min_y_i:max_y_i,min_x_i:max_x_i]
	
	julian_day_start = 1
	#julian_day_end = ndays #this should be determined by the length of the climate data stack
	julian_day_end = 4 #this should be determined by the length of the climate data stack
	nhours = (julian_day_end-julian_day_start) * 24
	hourly_temp = np.zeros((nhours,dim[0],dim[1])) #storage array for our output
	hours = np.zeros((nhours,dim[0],dim[1]))
	
	filename = 'K:\\NASA_data\\solar_out\\%s_snsrss.nc'%(year)
	sp_ds = nc.Dataset(filename)
	
	#need to find the high and low points and divide into 24 periods?
	t_dmin_prev = np.ones(dim) * (julian_day_start-1) #initialize the first minimum
	for i in range(julian_day_start, julian_day_end):
		#get the sun position arrays
		sunrise = ds_unix_time_convert(sp_ds, 'sunrise', i-1, year) /24 #in days
		transit = ds_unix_time_convert(sp_ds, 'transit', i-1, year) /24
		sunset = ds_unix_time_convert(sp_ds, 'sunset', i-1, year) /24
		sunrise_next = (ds_unix_time_convert(sp_ds, 'sunrise', i, year) /24) + 1
		day_hour = i-1
		t_dmax = (sunset - transit)/2 + transit + day_hour #calculate the time when maximum temperature is reached in hours
		t_dmin = sunrise + day_hour #calculate the time when minimum temperature is reached
		t_dmin_next = sunrise_next + day_hour
		#so we can make a time array that starts at the t_dmin_prev and then goes to the last t_dmin_next. 
		#but we would like to start there and then go to each hourly increment (1/24) of a day...
		#this is not the most easy to perform maybe?
		#so get the decimal position, then count up to mext?
		bound_time = np.tensordot(np.linspace(0,1,25),(t_dmin_next-t_dmin_prev),axes=0)  + t_dmin_prev ##problem here
		#first off, maybe change all the values that are not the first nor last data point into whole integer values
		#that way we can reference them?
		for t_hour in range(len(bound_time)): #twenty four hour day division
			index = day_hour * 24 + t_hour ##also problem
			print(index)
			#current_julian_hour = day_hour + t_hour/24 * np.ones(dim)
			current_julian_hour = bound_time[t_hour]
			current_P = cos_transform_array(P_max[i-1], P_min[i-1], t_dmin, t_dmax, t_dmin_next, current_julian_hour)
			if ((i != julian_day_start) and (t_hour == 0)): #for the first new day
				hourly_temp[index] = (current_P + last_P)/2 # take the average between two days
				hours[index] = current_julian_hour
			elif (t_hour == len(bound_time)-1): #last hour
				last_P = current_P
			else:
				hourly_temp[index] = current_P
				hours[index] = current_julian_hour
		t_dmin_prev = bound_time[-1]
	h_time = hours.mean(axis =(1,2))
	t_hourly = hourly_temp.mean(axis=(1,2))
	plt.plot(h_time, t_hourly)
	##still not working right....@t.chang 09.02.2015

if __name__ == "__main__":
	main()
	
	