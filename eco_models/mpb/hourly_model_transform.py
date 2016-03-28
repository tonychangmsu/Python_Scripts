#Title: hourly_model_transform.py
#Author: Tony Chang
#Date: 8/6/2015
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
# t is the julien hour, t_dmin is the julien hour for day when the minimum temperature is reached,
# t_dmax is the julien hour for day when the maximum temperatuer is reached
# this will require an understanding the latitude and potentially the elevation 

# furthermore there is a requirement for knowing the sunrise, sunset, and solar noon times which are determined by the
# latitude/longitude and Julian day. 
# This code has been implemented, but is in need of refinement. 
# there are 1440 minutes in the day

# so let's try is for a today July 8 julian hours

import numpy as np
import matplotlib.pyplot as plt


import os
import netCDF4 as nc
os.chdir('E:\\TOPOWX')
import geotool as gt
import util_dates as utld
import gdal as gdal
import time
#set the working directory to one containing twx_sumry_v2


#new code 	07.30.2015 @tchang and 
#			08.06.2015 @tchang
#calculate the solar radiation from Kumar et al 1997 Modeling topographic variation in solar radiation in a GIS environment
#first we need to obtain the DEM data.

def topo_extract():
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

def solar_altitude_angle(declination, hour_angle, latitude):
	#from Kumar et al 1997 Eq (2)
	rad_dec = np.radians(declination)
	rad_lat = np.radians(latitude)
	rad_hour = np.radians(hour_angle)
	saa = np.arcsin((np.sin(rad_lat)* np.sin(rad_dec)) + (np.cos(rad_dec)*np.cos(rad_hour)))
	return(np.degrees(saa))

def solar_azimuth_angle(declination, hour_angle, altitude_angle):
	#from Kumar et al 1997 Eq (3)
	rad_dec = np.radians(declination)
	rad_hour = np.radians(hour_angle)
	rad_alt = np.radians(altitude_angle)
	azi = np.arcsin((np.cos(rad_dec)*np.sin(rad_hour))/np.cos(rad_alt))
	return(np.degrees(azi))
	
def solar_declination(julian_day): 
	C1 = 0.006918
	C2 = 0.399912
	C3 = 0.070257
	C4 = 0.006758
	C5 = 0.000907
	C6 = 0.002697
	C7 = 0.001480
	gamma = 2*np.pi/365*(julian_day-1)
	declination = C1 - C2*np.cos(gamma) + C3*np.sin(gamma)- C4*np.cos(2*gamma) + C5*np.sin(2*gamma) - C6*np.cos(3*gamma) + C7*np.sin(3*gamma)
	#from Spencer, J.W. 1971 Fourier series representation of the position of the sun
	#declination = np.radians(23.45) * np.sin(np.radians(360/365*(284+julian_day))) #less accurate calculation from Duffie and Beckman 1991
	#alternatively Michalsky, J.J. 1988 also wrote an algorithm from the Astronomical Almanac. This might be used if one is motivated to use
	#it, however it only applies to 1950-2050.
	return(np.degrees(declination))

#############################
#new today 08.04.2015 @tchang
#derived from Seidelmann et al 2000

def local_standard_time_meridian(delta_gmt):
	#knowing the regions difference from local time from Greenwich Mean Time in hours
	#returns the local standard time meridian (LSTM) in the units radian hours
	return(15*delta_gmt)
	
def time_correction(longitude, lstm, eot):
	#returns the time correction factor (in minutes) that account for the variation of the local solar time(LST)
	#within a given time zone due to the longitude variations within the time zone and also incorporates the 
	#equation of time
	time_correction_factor = (4*(longitude - lstm)) + eot
	return(time_correction_factor)

def equation_of_time(julian_day):
	D = np.radians(360)*((julian_day-81)/365)
	eqt = 9.87 * np.sin(2*D) - 7.53*np.cos(D) - 1.5*np.sin(D)
	return(eqt)
	
def local_solar_time(local_time, tc):
	#returns the local solar time (LST) (where solar noon represents when the sun is directly 90 degrees of current position)
	#transform the local time array to match the time correction factor grid
	if (len(local_time)!=1):
		local_time_array = np.tensordot(local_time, np.ones(np.shape(tc)), axes = 0)
		local_solar_time = local_time_array + tc/60
	else:
		local_solar_time = local_time + tc/60
	return(local_solar_time)

def solar_time_grid(jd, lon_grid, time_step):
	zonal_delta_gmt = time.localtime().tm_hour - time.gmtime().tm_hour #difference between local time (Montana) versus greenwich mean time
	local_meridian = local_standard_time_meridian(zonal_delta_gmt)
	eqt = equation_of_time(jd)
	time_correction_factor = time_correction(lon_grid, local_meridian, eqt)
	ast = local_solar_time(time_step, time_correction_factor) #apparent solar time
	return(ast) #returns a grid of the apparent solar time for the specified time increments over the lat/lon grid

def hour_angle(local_solar_time):
	#returns the hour angle (HRA) given the local solar time of where the sun is positioned
	#by definition, the hour angle is 0 deg at solar noon. Since the Earth rotates at approximately 15 deg per hour, each hour
	#away from solar noon corresponds to an angular motion of the sun in the sky of 15 deg. In the morning the hour angle is 
	#negative and in the afternoon the hour angle is positive. This will be returned in degrees.
	return(15*(local_solar_time-12))

def zero_hour_angle(lat_grid, declination):
	# this is the hour angle when solar altitude is zero
	# hsr = np.arccos(-1*np.tan((lat_grid))*np.tan((declination)))) 
	# since the original equation fails to take into account that the sun is a disk not a point, 
	# so we need to allow for the sunrise and sunset to account for that
	h_o = np.radians(-0.83)
	#d_sun = 0.53
	lat_rad = np.radians(lat_grid)
	dec_rad = np.radians(declination)
	hsr = np.arccos((np.sin(h_o)-(np.sin(lat_rad)*np.sin(dec_rad)))/((np.cos(lat_rad)*np.cos(dec_rad))))
	return(np.degrees(hsr)) #this is the hour angle when solar altitude is zero
	
def sunset_sunrise_solver(hour_ang, hsr, grid_dimension, thr = 0.5):
	#generates a set of tuples that denote the sunrise, solar noon, and sunset times. 
	#We will make an assumption that the maximum temperature occurs about halfway between solar noon and sunset. 
	#Minimum temperature could occur halfway between sunset and sunrise of the next day. 
	sr_index = hour_ang + hsr #sunrise #- hsr is sunrise
	ss_index = hour_ang - hsr #sunset  #+ hsr is sunset
	sr_i = np.where(np.abs(sr_index) <= thr)
	ss_i = np.where(np.abs(ss_index) <= thr)
	sr = np.zeros(grid_dimension) #place holder
	sr[sr_i[1],sr_i[2]]=sr_i[0] #fill in the sr array with the minute of sunrise
	ss = np.zeros(grid_dimension)
	ss[ss_i[1],ss_i[2]]=ss_i[0] #fill in the ss array with the minute of sunset
	return(sr, ss)

def solar_noon(hour_ang, grid_dimension, thr = 0.5):	
	sn_i = np.where(np.abs(hour_ang) <= thr)
	sn = np.zeros(grid_dimension)
	sn[sn_i[1],sn_i[2]]=sn_i[0] #fill in the ss array with the minute of sunset
	return(sn)	
	
def sunrise_sunset(jd, lat_grid, lon_grid, time_step, grid_dimension):
	local_solar_grid = solar_time_grid(jd, lon_grid, time_step)
	hour_ang = hour_angle(local_solar_grid)
	declination = solar_declination(jd)
	solar_alt = solar_altitude_angle(declination, hour_ang, lat_grid) 
	solar_azi = solar_azimuth_angle(declination, hour_ang, solar_alt)
	hsr = zero_hour_angle(lat_grid, declination)
	sunrise, sunset = sunset_sunrise_solver(hour_ang, hsr, grid_dimension)
	noon = solar_noon(hour_ang, grid_dimension)
	return(sunrise, sunset, noon)
	
##########################################################################################

#seems to work well enough. just need to write a couple functions to calculate the t_dmax and t_dmin 
#given elevation, aspect, and latitude/longitude. Then apply to a grid. 
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
lat = nc_ds.variables['lat'][min_y_i:max_y_i]
lon = nc_ds.variables['lon'][min_x_i:max_x_i]
lon_grid,lat_grid = np.meshgrid(lon, lat)
grid_dimension = (len(lat), len(lon))
time_step = np.linspace(0,24,1440) #total number of minutes in a day starting at 12:00am to solve sunrise and sunset

jd = 1 #use this as the counter to solve for the sunset, sunrise, solar noon values
sunrise, sunset, noon = sunrise_sunset(jd, lat_grid, lon_grid, time_step, grid_dimension)

#Everything seems to be about 1 hour off. Could be associated to the Meridian time?
#so this looks pretty good, we have all the major components and now only have to gather this into nice functions rather than raw code.
#we might still consider the eccentricity of the Earth since that is changing in order to associate the time with the julian dates. 
#I'll think about it but for now, these values will have about 15 min of error here and there, but may be fine, since we will be
#considering growth every single hour. 