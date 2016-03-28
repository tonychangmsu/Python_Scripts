#Title: solar_calc.py
#Author: Tony Chang
#Abstract: Implementation of the NREL Solar Position Algorithm applied to a grid level to calculate sunrise, solar noon,
# and sunset for application of hourly temperature transformation. (Reda and Andreas 2008)
#Date: 08.11.2015

import datetime

# scientific python add-ons
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# finally, we import the pvlib library
import pvlib
from pvlib.location import Location

import netCDF4 as nc
import gdal as gdal

def topoExtract():
	#since shape of the GCM is different from the PRISM extent, a modified grid is required if gcm parameter is set to 'y'
	aspectPath = "E:\\gye_topo\\aspect_800m.tif"   
	slopePath = "E:\\gye_topo\\slope_800m.tif"   
	elevPath = "E:\\gye_topo\\dem_800m.tif"   
	ds = gdal.Open(aspectPath)
	aspect = np.array(ds.GetRasterBand(1).ReadAsArray())
	ds = gdal.Open(slopePath)
	slope = np.array(ds.GetRasterBand(1).ReadAsArray())
	ds = gdal.Open(elevPath)
	elev = np.array(ds.GetRasterBand(1).ReadAsArray())
	ds = None #close files
	return(aspect[:,:], slope[:,:], elev[:,:]) 
	
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
lat = nc_ds.variables['lat'][min_y_i:max_y_i]
lon = nc_ds.variables['lon'][min_x_i:max_x_i]
lon_grid,lat_grid = np.meshgrid(lon, lat)
grid_dimension = (len(lat), len(lon))
aspect, slope, elev = topoExtract()
#here we can loop for all months/days/years of interest and see how long that takes?
#could be performed over a weekend....with clearing of data everyday. 
#again, all we need to do is generate a netCDF4 file with 3 variables, sunrise, sunset, and solar noon
#we can generate one netCDF4 for each year with the time variable as julian day from 0 to 364
#this way we will have matching netCDF4 with the climate data
m = 8
d = 12
y = 2015
times = pd.date_range(start=datetime.datetime(y, m, d), end = datetime.datetime(y,m,d+1), freq='1Min')

locations = []
solar_noon = np.zeros((len(lat),len(lon)))
sunrise_grid = np.zeros((len(lat),len(lon)))
sunset_grid = np.zeros((len(lat),len(lon)))

lats = np.reshape(lat_grid, (len(lat)*len(lon)))
longs = np.reshape(lon_grid, (len(lat)*len(lon)))
elevs = np.reshape(elev, (len(lat)*len(lon)))

saves = np.zeros(np.shape(lats))
#specify the tz when declaring time...
times = pd.date_range(start=datetime.datetime(y, m, d), end = datetime.datetime(y,m,d), tz= 'US/Mountain')

def unix_time_convert(time):
	utcday  = time.tz_convert('UTC')
	unixtime = utcday.astype(np.int64)/10**9
	return(unixtime)
	
#now we can use the solar position functions with the unixtime (date)
unixtime2 = unix_time_convert(times)
delta_t = 67.0
pressure = 101325 / 100
temp = 12
atmos_refract = 0.5667
numthreads = 1



spa.solar_position_numpy(unixtime
def solar_times(location, time):
	#current_location = Location(lat, lon, 'America/Denver', elev, 'GYE') 
	out = pvlib.solarposition.get_sun_rise_set_transit(time, location, how='numba', numthreads=3)
	solar_noon = out.transit[0].hour + out.transit[0].minute/60
	return(solar_noon)
'''
locations = []
for i in range(len(saves)):
	locations.append(Location(lats[i], longs[i], 'America/Denver', elevs[i], 'GYE') )
	#saves[i] = solar_times(lats[i], longs[i], elevs[i], times[0])
	print(i)
locations = np.array(locations)
'''
for i in range(len(saves)):
	saves[i] = solar_times(locations[i], times)
	print(i)
#still too slow, need to vectorize the equations so that I can get all the solutions at once. 
#tchang
def solar_noon(lat, lon, elev):
	#calculates solar noon given a lat, lon, and elevation
	#all these functions lag in performance because we must instantiate the object for location and times every time....
	current_location = Location(lat, lon, 'America/Denver', elev, 'GYE') 
	times = pd.date_range(start=datetime.datetime(y, m, d), end = datetime.datetime(y,m,d+1), freq='1Min')
	spaout = pvlib.solarposition.spa_python(times, current_location)
	sn_t = spaout.apparent_elevation.idxmax().hour + spaout.apparent_elevation.idxmax().minute/60
	return(sn_t)

#running a loop on this should take about 3 hours per day evaluated....this is too slow. need a better solution.
#calculations would take 8 years to produce a solution for all days and all cells. 

vec_solar_noon = np.vectorize(solar_noon)
out = vec_solar_noon(lats, longs, elevs)
for i in range(len(lat)):
	for j in range(len(lon)):
		current_location = Location(lat[i], lon[j], 'America/Denver', elev[i][j], 'GYE')
		locations.append(current_location)
		spaout = pvlib.solarposition.spa_python(times, current_location)
		#get the solar noon values first to check on it
		sn_t = spaout.apparent_elevation.idxmax().hour + spaout.apparent_elevation.idxmax().minute/60
		solar_noon[i][j] = sn_t
		
		


