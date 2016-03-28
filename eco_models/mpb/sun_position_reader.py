"""
Title: sun_position_reader.py
Author: Tony Chang
Date: 08.25.2015
Abstract: 	Read the netCDF4 file of solar noon, sunset, and sunrise and convert
			from unixtime array to hours (local time)

"""

import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime as datetime
import pytz as pytz
import calendar as calendar

def unix_time_convert(year, month, day, local_tz):
	#takes in year, month, day and time zone and returns time since Epoch (unixtime) in UTC
	date = '%s/%s/%s'%(year, month, day)
	naive = datetime.datetime.strptime("%s"%(date), "%Y/%m/%d")
	local_dt = local_tz.localize(naive) #assign the time zone
	utc_dt = local_dt.astimezone(pytz.utc) 
	#this is correct (6 hour difference)
	unixtime_local = calendar.timegm(local_dt.timetuple())
	unixtime_utc = calendar.timegm(utc_dt.timetuple()) 
	return(unixtime_utc)

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
	
#now we have the month and date convert to unixtime at 00:00:00
#subtract this unixtime_utc from the data arrays
#divide data by 3600 to get hours
hours_per_second = 1/(60*60)
#my initial idea is to get the date at 0:00:00 from the netCDF and then subtract that from the 
#values and then attain the hour from there. Since we will get seconds back and just need to convert

local_tz = pytz.timezone("US/Mountain")
year = 1948
readspace = 'K:\\NASA_data\\solar_out\\'
readfile = '%s_snsrss.nc'%(year)
ds = nc.Dataset('%s%s'%(readspace, readfile))
i = 39
f_sr = ds_unix_time_convert(ds, 'sunrise', i+1, year)
f_sn = ds_unix_time_convert(ds, 'transit', i+1, year)
f_ss = ds_unix_time_convert(ds, 'sunset', i+1, year)
