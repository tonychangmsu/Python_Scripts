#Title: hourly_model.py
#Author: Tony Chang
#Date: 7/8/2015
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
# sunrise is at 5:42 am
# sunset is at 21:17 pm
# there are 1440 minutes in the day

# so let's try is for a today July 8 julien hours
# 189 day and each hour is equal to 1/24

import numpy as np
import matplotlib.pyplot as plt

# julien_day = 189
# julien_hour = np.linspace(189, 190, 25)

# P_max = 26
# P_min = 10

# P = []

# minutes = np.linspace(0,1, 1441)
# last_sunset = 21*60 + 17
# sunrise = 5*60 + 42
# sunset = 21*60 + 17
# sunrise_next = 5*60 + 42

# t_dmax = julien_day + minutes[round(sunrise+((sunset - sunrise)/2))] #around 1:30 pm is the maximum 
# t_dmin = julien_day + minutes[round(sunset+((sunset - sunrise_next)/2)-1440)]
# t_dmin_next = julien_day + 1 + minutes[round(sunset+((sunset - sunrise_next)/2)-1440)]

# for i in julien_hour:
	# P.append(Pt(P_max, P_min, t_dmin, t_dmax, t_dmin_next, i))

# def Pt(P_max, P_min, t_dmin, t_dmax, t_dmin_next, t):
	# #if (t<t_dmin):
	# #	P_t = 0.5*(P_max + P_min) + 0.5*(P_max - P_min) * cos(((pi*(t-t_dmax))/((t_dmin-1)-t_dmax))) #not right yet....
	# #if ((t>t_dmin_prev) and (t<t_dmax)):
	# if (t<t_dmax):
		# P_t = 0.5*(P_max + P_min) + 0.5*(P_max - P_min) * cos(pi + ((pi*(t-t_dmin))/(t_dmax-t_dmin)))
	# elif (t>t_dmax):
		# P_t = 0.5*(P_max + P_min) + 0.5*(P_max - P_min) * cos(((pi*(t-t_dmax))/((t_dmin_next)-t_dmax)))
	# return(P_t)	

# Addition of the McCune and Keon 2002 heat load functions
"""While these equations have proven useful, their limitations must be recognized. They do not give account to
cloud cover, regional differences in the atmospheric coefficient, and shading by adjacent topography. -McCune and Keon 2002"""
'''
def folded_aspect(aspect): #takes in aspect in degrees
    af = np.ma.masked_array(aspect, aspect==-1)
    af.fill_value = np.nan     #maybe adjust this so that all aspect values of -1 do not get calculated
    af = af.filled()
    af = abs(math.pi-abs((np.radians(af))-(math.pi*5/4)))    
    return(af) #returns the folded aspect in radians format
	
def solar_dec():
    #calculates the solar declination angle for a given latitude and month 
    #We use the solar declination angle at noon on the 15th of the given month    
    daypmon = np.array([31,28,31,30,31,30,31,31,30,31,30,31]) #number of days per month, not considering leap year
    D = daypmon-15
    ffthday = []
    for i in range(len(daypmon)):
        if i == 0:
            ffthday.append(daypmon[i]-D[i])
        else:
            ffthday.append(ffthday[i-1]+D[i-1]+daypmon[i]-D[i])
    ffthday = np.array(ffthday)
    sd = -23.45*(math.pi/180)* np.cos((2*math.pi)*(ffthday+10)/365) 
    return(sd)

def day_length(sd, lat):  
    av = 0.2618 #angular velocity of the Earth's rotation (rad/hr)        
    lat = np.radians(lat) #convert to radians    
    dl = []    
    for i in range(len(sd)):
        dl.append((2*np.arccos(-np.tan(sd[i])*np.tan(lat)))/av)
    dl = np.array(dl) #day length
    return(dl)
    
def heat_load(slope, aspect, lat):    
    af = (folded_aspect(aspect))
    slope = np.radians(slope) #convert to radians
    hl = 0.339 + 0.808*(np.cos(lat)*np.cos(slope)) - 0.196*(np.sin(lat)*np.sin(slope)) - 0.482*(np.cos(af)*np.sin(slope))
    nanIndex = np.isnan(hl) #locate all the nan values (no aspect)
    fillnan = hl[nanIndex] #index of the nan values
    fillnan = 0.339 + 0.808*(np.cos(lat[nanIndex])*np.cos(slope[nanIndex])) - 0.196*(np.sin(lat[nanIndex])*np.sin(slope[nanIndex])) #replace nan values with HL equation without aspect load
    hl[nanIndex] = fillnan #insert values into nan elements
    return(HL)
'''
# #now we need to adapt this to consider more days
# julien_day_start = 189
# julien_day_end = 191
# t_dmin_prev = julien_day_start #initialize the first minimum
# minutes = np.linspace(0,1, 1441)
# #each hour unit is 1/24 or 0.041666
# P = []
# hours = []
# for d in range(julien_day_start, julien_day_end):
	# last_sunset = 21*60 + 17
	# sunrise = 5*60 + 42
	# sunset = 21*60 + 17
	# sunrise_next = 5*60 + 42
	# t_dmax = d + minutes[round(sunrise+((sunset - sunrise)/2))] #around 1:30 pm is the maximum 
	# t_dmin = d + minutes[round(sunset+((sunset - sunrise_next)/2)-1440)]
	# t_dmin_next = d + 1 + minutes[round(sunset+((sunset - sunrise_next)/2)-1440)]
	
	# #julien_hour = np.linspace(d,d+1,25) #need to redefine this so the bounds are by the next julien day minimum temperature
	# julien_hour = np.arange(t_dmin_prev, t_dmin_next, (1/24))

	# #initialize the min and max temperatures and the timing of them

#Title: hourly_model.py
#Author: Tony Chang
#Date: 7/8/2015
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
# sunrise is at 5:42 am
# sunset is at 21:17 pm
# there are 1440 minutes in the day

# so let's try is for a today July 8 julien hours
# 189 day and each hour is equal to 1/24

import numpy as np
import matplotlib.pyplot as plt
import timeit 
# julien_day = 189
# julien_hour = np.linspace(189, 190, 25)

# P_max = 26
# P_min = 10

# P = []

# minutes = np.linspace(0,1, 1441)
# last_sunset = 21*60 + 17
# sunrise = 5*60 + 42
# sunset = 21*60 + 17
# sunrise_next = 5*60 + 42

# t_dmax = julien_day + minutes[round(sunrise+((sunset - sunrise)/2))] #around 1:30 pm is the maximum 
# t_dmin = julien_day + minutes[round(sunset+((sunset - sunrise_next)/2)-1440)]
# t_dmin_next = julien_day + 1 + minutes[round(sunset+((sunset - sunrise_next)/2)-1440)]

# for i in julien_hour:
	# P.append(Pt(P_max, P_min, t_dmin, t_dmax, t_dmin_next, i))

def Pt(P_max, P_min, t_dmin, t_dmax, t_dmin_next, t):
	#if (t<t_dmin):
	#	P_t = 0.5*(P_max + P_min) + 0.5*(P_max - P_min) * cos(((pi*(t-t_dmax))/((t_dmin-1)-t_dmax))) #not right yet....
	#if ((t>t_dmin_prev) and (t<t_dmax)):
	if (t<t_dmax):
		P_t = 0.5*(P_max + P_min) + 0.5*(P_max - P_min) * cos(pi + ((pi*(t-t_dmin))/(t_dmax-t_dmin)))
	elif (t>t_dmax):
		P_t = 0.5*(P_max + P_min) + 0.5*(P_max - P_min) * cos(((pi*(t-t_dmax))/((t_dmin_next)-t_dmax)))
	return(P_t)	
	
#now we need to adapt this to consider more days
#julien_day_start = 189
#julien_day_end = 195
julien_day_start = 1
julien_day_end = 3
ndays = julien_day_end-julien_day_start
t_dmin_prev = julien_day_start #initialize the first minimum
minutes = np.linspace(0,1, 1441)
#each hour unit is 1/24 or 0.041666
P = []
hours = []

P_max = 26
P_min = 10
nx = 471
ny = 504 
#check the speed with a single year of data at the gye scale
P_max_array = np.random.normal(P_max, 5, (ndays,nx,ny))
P_min_array = np.random.normal(P_min, 3, (ndays,nx,ny))
j = 0
tic = timeit.default_timer()

for d in range(julien_day_start, julien_day_end):
	#initialize the min and max temperatures and the timing of them
	P_max = P_max_array[j]
	P_min = P_min_array[j]
	last_sunset = 21*60 + 17
	sunrise = 5*60 + 42
	sunset = 21*60 + 17
	sunrise_next = 5*60 + 42
	
	t_dmax = d + minutes[round(sunrise+((sunset - sunrise)/2))] #around 1:30 pm is the maximum 
	t_dmin = d + minutes[round(sunset+((sunset - sunrise_next)/2)-1440)]
	t_dmin_next = d + 1 + minutes[round(sunset+((sunset - sunrise_next)/2)-1440)]
	
	#julien_hour = np.linspace(d,d+1,25) #need to redefine this so the bounds are by the next julien day minimum temperature
	julien_hour = np.arange(t_dmin_prev, t_dmin_next, (1/24))
	for i in range(len(julien_hour)):
		currentP = Pt(P_max, P_min, t_dmin, t_dmax, t_dmin_next, julien_hour[i]) 
		if ((d != julien_day_start) and (i == 0)): #if this is the first hour, we need to reference the last day
			P.append((currentP + lastP)/2) # take the average between two days
			hours.append(julien_hour[i])
		elif (i == len(julien_hour)-1): #last iteration
			lastP = currentP
		else: #other iterations just save the hour and temperature
			P.append(currentP)
			hours.append(julien_hour[i])
	t_dmin_prev = julien_hour[-1] #save for the next iteration of the loop
	j += 1
P = np.array(P)
hours = np.array(hours)
#hours = np.array(julien_hour)
toc = timeit.default_timer()
print("The total process time is %0.1f" %(toc-tic))
plt.plot(hours, P.mean(axis = (1,2)))	#take the average for the domain to examine the average hourly temperature
plt.xlabel('Julien Day')
plt.ylabel('Modeled phloem tempuratre ($^oC)$')
mem = P.nbytes/1e6 #how many megabytes this array takes up, looks like 16 GB.
#if there are 64 years, then we are talking 1024 GB or over 1 terabyte to store all this data!!
# in any case lets try to apply it to a single year from the GYE TOPOWX dataset..

import os
import netCDF4 as nc
os.chdir('E:\\TOPOWX')
import geotool as gt
import util_dates as utld
import gdal as gdal
import time
#set the working directory to one containing twx_sumry_v2

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

#new code 07.30.2015 @tchang
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

def hour_to_min(hours):
	return(hours*60)
	
def apparent_solar_time(hours, long, eq_of_time):
	minutes = hour_to_min(hours)
	lstm = 15 * np.int(long/15) #local standard time meridian
	ast = minutes + 4 * (lstm-long)+eq_of_time
	return(ast)

def equation_of_time(julian_day):
	D = np.radians(360)*((julian_day-81)/365)
	eqt = 9.87 * np.sin(2*D) - 7.53*np.cos(D) - 1.5*np.sin(D)
	return(eqt)
	
def hour_angle(ast):
	H = np.radians((ast - 720) / 4)
	return(H)
	
#############################
#new today 08.04.2015 @tchang
def lstm(delta_gmt):
	#knowing the regions difference from local time from Greenwich Mean Time in hours
	#returns the local standard time meridian (LSTM) in the units radian hours
	return(15*delta_gmt)
	
def tc(longitude, lsm, eot):
	#returns the time correction factor (in minutes) that account for the variation of the local solar time(LST)
	#within a given time zone due to the longitude variations within the time zone and also incorporates the 
	#equation of time
	time_correction_factor = (4*(longitude - lsm)) + eot
	return(time_correction_factor)

def lst(local_time, tc):
	#returns the local solar time (LST) (where solar noon represents when the sun is directly 90 degrees of current position)
	#transform the local time array to match the time correction factor grid
	if (len(local_time)!=1):
		local_time_array = np.tensordot(local_time, np.ones(np.shape(tc)), axes = 0)
		local_solar_time = local_time_array + tc/60
	else:
		local_solar_time = local_time + tc/60
	return(local_solar_time)

def hra(lst):
	#returns the hour angle (HRA) given the local solar time of where the sun is positioned
	#by definition, the hour angle is 0 deg at solar noon. Since the Earth rotates at approximately 15 deg per hour, each hour
	#away from solar noon corresponds to an angular motion of the sun in the sky of 15 deg. In the morning the hour angle is 
	#negative and in the afternoon the hour angle is positive. This will be returned in degrees.
	return(15*(lst-12))

def dec(d):
	#returns the declination angle from d the julian day of year
	delta = 23.45 * np.sin(np.radians((360/365)*(d-81)))
	return(delta)
	
###############
#okay lets try for january 1st and june 30
lat = nc_ds.variables['lat'][min_y_i:max_y_i]
lon = nc_ds.variables['lon'][min_x_i:max_x_i]
lon_grid,lat_grid = np.meshgrid(lon, lat)
jd = 1 #we will iterate on this for each day of the year. 
hours = np.linspace(0,24,1440) #total number of minutes in a day starting at 12:00am
zonal_delta_gmt = time.localtime().tm_hour - time.gmtime().tm_hour #difference between local time (Montana) versus greenwich mean time
lstm_try = lstm(zonal_delta_gmt)
eqt = equation_of_time(jd)
time_correction = tc(lon_grid, lstm_try, eqt)
local_solar = lst(hours, time_correction)
hour_ang = hra(local_solar)
##not done yet, need to calculate the altitude angle to get actual sunrise and sunset...
declination = solar_declination(jd)
solar_alt = solar_altitude_angle(declination, hour_ang, lat_grid) 
solar_azi = solar_azimuth_angle(declination, hour_ang, solar_alt)
#what I'd like to have is a grid of tuples that denote the sunrise, solar noon, and sunset times. If I have these values I can make an assumption 
#that the maximum temperature occurs about halfway between solar noon and sunset. Minimum temperature could occur halfway between sunset
#and sunrise of the next day. 

#we need 
hsr = np.degrees(np.arccos(-1*np.tan(np.radians(lat_grid))*np.tan(np.radians(declination)))) #this is the hour angle when solar altitude is zero
#this equation fails to take into account that the sun is a disk not a point, so we need to allow for the sunrise and sunset to account for that
#2 coefficients
h_o = -0.83
d_sun = 0.53
hsr = np.degrees(np.arccos((np.sin(np.radians(h_o))-(np.sin(np.radians(lat_grid))*np.sin(np.radians(declination))))/((np.cos(np.radians(lat_grid))*np.cos(np.radians(declination)))))) #this is the hour angle when solar altitude is zero
sr_index = np.round((hour_ang + hsr), decimals =0) #sunrise #- hsr is sunrise
ss_index = np.round((hour_ang - hsr), decimals =0) #sunset  #+ hsr is sunset
#make an array for sunrise
sr_i = np.where(sr_index==0)
ss_i = np.where(ss_index==0)
sr = np.zeros(np.shape(lat_grid)) #place holder
sr[sr_i[1],sr_i[2]]=sr_i[0] #fill in the sr array with the minute of sunrise
ss = np.zeros(np.shape(lat_grid))
ss[ss_i[1],ss_i[2]]=ss_i[0] #fill in the ss array with the minute of sunset

#still need solar noon, where the hour angle is equal to 0
sn_i = np.where(np.round(hour_ang)==0)
sn = np.zeros(np.shape(lat_grid))
sn[sn_i[1],sn_i[2]]=sn_i[0] #fill in the ss array with the minute of sunset

#so this looks pretty good, we have all the major components and now only have to gather this into nice functions rather than raw code.
#we might still consider the eccentricity of the Earth since that is changing in order to associate the time with the julian dates. 
#I'll think about it but for now, these values will have about 15 min of error here and there, but may be fine, since we will be
#considering growth every single hour. 
##################################
#1: so first thing is the calculate the solar altitude angle for every hour for every day of the year#note to obtain the latitude and longitude record as well
lat = nc_ds.variables['lat'][min_y_i:max_y_i]
lon = nc_ds.variables['lon'][min_x_i:max_x_i]
lon_grid,lat_grid = np.meshgrid(lon, lat)
julian_time = (np.array(nc_ds.variables['time'][:])) + 0.5 #add 0.5 to get the Julien day since TOPOWX uses the julian hour at 12:00pm
hours = np.arange(0,24)

julian_grid = np.tensordot(julian_time, np.ones((len(lat), len(lon))), axes = 0) #create an time matrix for all points on domain 

zonal_delta_gmt = time.localtime().tm_hour - time.gmtime().tm_hour #difference between local time (Montana) versus greenwich mean time

eoq = equation_of_time(julian_grid)


#now we have the julian hour for all the point we require
#first solve for solar declination
declination = solar_declination(julien_time)


saa = solar_altitude_angle(declination, ast, latitude)
 
#2: using these solar altitude angles, we must denote every single day, where the sunrise hour is and sunset hour is
#	This has been noted as the solar hour when solar altitude is either 0 (Kumar et al 1997) or -0.8333 (Fleming et al 1995)

aspect,slope,elevation = topo_extract()


startyear = 1948
endyear = 1949
var = ['tmin','tmax']
full_data = []
for v in var:
	dataset = []
	for year in range(startyear, endyear):
		filename = '%s%s\\%s_%s.nc' %(workspace, v, v, str(year))
		nc_ds = nc.Dataset(filename)
		days = utld.get_days_metadata_dates(nc.num2date(nc_ds.variables['time'][:], units=nc_ds.variables['time'].units))
		daily_dataset = nc_ds.variables[v][:,min_y_i:max_y_i,min_x_i:max_x_i]
	full_data.append(daily_dataset)
	
days_in_year = np.round(nc_ds.variables['time'][-1]) #find out if it is a leap year

P_min_array = full_data[0]
P_max_array = full_data[1]

j = 0
tic = timeit.default_timer()
P = []
hours = []
for d in range(julien_day_start, julien_day_end):
	#initialize the min and max temperatures and the timing of them
	P_max = P_max_array[j]
	P_min = P_min_array[j]
	last_sunset = 21*60 + 17
	sunrise = 5*60 + 42
	sunset = 21*60 + 17
	sunrise_next = 5*60 + 42
	
	t_dmax = d + minutes[round(sunrise+((sunset - sunrise)/2))] #around 1:30 pm is the maximum 
	t_dmin = d + minutes[round(sunset+((sunset - sunrise_next)/2)-1440)]
	t_dmin_next = d + 1 + minutes[round(sunset+((sunset - sunrise_next)/2)-1440)]
	
	#julien_hour = np.linspace(d,d+1,25) #need to redefine this so the bounds are by the next julien day minimum temperature
	julien_hour = np.arange(t_dmin_prev, t_dmin_next, (1/24))
	for i in range(len(julien_hour)):
		currentP = Pt(P_max, P_min, t_dmin, t_dmax, t_dmin_next, julien_hour[i]) 
		if ((d != julien_day_start) and (i == 0)): #if this is the first hour, we need to reference the last day
			P.append((currentP + lastP)/2) # take the average between two days
			hours.append(julien_hour[i])
		elif (i == len(julien_hour)-1): #last iteration
			lastP = currentP
		else: #other iterations just save the hour and temperature
			P.append(currentP)
			hours.append(julien_hour[i])
	t_dmin_prev = julien_hour[-1] #save for the next iteration of the loop
	j += 1
P = np.array(P)
hours = np.array(hours)
#hours = np.array(julien_hour)
toc = timeit.default_timer()
print("The total process time is %0.1f" %(toc-tic))
plt.plot(hours, P.mean(axis = (1,2)))	#take the average for the domain to examine the average hourly temperature
plt.xlabel('Julien Day')
plt.ylabel('Modeled phloem tempuratre ($^oC)$')
#output looks good, but again the daily sunrise and sunset needs to be adjusted for each particular day, so that needs to be adjusted
mem = P.nbytes/1e6 #how many megabytes this array takes up.
#the idea would be to generate a new netCDF4 file for the hourly data with both variables such that we could just open them up.
	