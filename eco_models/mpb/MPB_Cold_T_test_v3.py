#Title: MPB_Cold_T_gridtest.py
#Author: Tony Chang
#Date: 02.04.2015
#Abstract: Testing the functionality of the MPB_coldT_model.py functions under a single year of
#gridded climate data


import MPB_coldT_model_v2 as mpb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc

#I want to get the data from TOPOWX annual tmin and tmax for the year of 1991, when survival was low in the Sawtooth National 
#Recreation area of central Idaho to confirm the model functionality

def leapYearCheck(year):
	if (year % 4) == 0:
		if (year % 100) == 0:
			if (year % 400) == 0:
				return(True)
			else:
				return(False)
		else:
			return(True)
	else:
		return(False)

def subsetClimate(nc_ds_filename, data_type, space_bounds, time_bounds):
	#subsets the netCDF4 dataset to the AOA
	#inputs the filename and the AOA
	#outputs the dataset for the start and end time of the time series
	nc_ds = nc.Dataset(nc_ds_filename)
	climate_out = nc_ds.variables[data_type][time_bounds[0]:time_bounds[1],space_bounds[1]:space_bounds[3],space_bounds[0]:space_bounds[2]]
	nc_ds.close()
	return(climate_out)
	
def getClimate(year, AOA=None, ws=None):
	#input the year of interest, the 
	#AOA area of analysis as [xmin, ymin, xmax, ymax]
	#and the workspace directory where climate data is stored
	#by default, the function will use GYE
	
	if ws == None:
		ws = 'E:\\TOPOWX\\annual\\' #define the workspace where the climate data is located
	var = ['tmin', 'tmax']
	years = [year,year+1] # because we are considering the dates from Aug 1 to the following Aug 1 we need 2 years of data
	
	tmin_file1 = '%s%s\\%s_%s.nc'%(ws,var[0], var[0],years[0])
	tmin_file2 = '%s%s\\%s_%s.nc'%(ws,var[0], var[0],years[1])
	tmax_file1 = '%s%s\\%s_%s.nc'%(ws,var[1], var[1],years[0])
	tmax_file2 = '%s%s\\%s_%s.nc'%(ws,var[1], var[1],years[1])
	tmin_filenames = [tmin_file1, tmin_file2]
	tmax_filenames = [tmax_file1, tmax_file2]
	#use the GYE bounds to reduce the size of area needed
	csize = 0.00833333333
	if AOA == None:
		#if no AOA set, using GYE
		xmax = -108.19583334006; xmin = -112.39583333838; ymin = 42.270833326049996; ymax = 46.19583332448 # GYE bounds
		AOA = [xmin, ymin, xmax, ymax] #specify the bounds for the FIA data
	
	nc_ds = nc.Dataset(tmin_file1)
	max_x_i = np.where(nc_ds.variables['lon'][:]>=AOA[2])[0][0] 
	min_x_i = np.where(nc_ds.variables['lon'][:]>=AOA[0])[0][0]
	min_y_i = np.where(nc_ds.variables['lat'][:]<=AOA[3])[0][0]
	max_y_i = np.where(nc_ds.variables['lat'][:]<=AOA[1])[0][0]
	
	space_bounds = [min_x_i, min_y_i, max_x_i, max_y_i] #sets the spatial extent for the data
	
	nc_ds.close()
	
	#need to check for leap year (feb 29?) for both years to set the temporal boundaries
	#if it is a leap year the Julien date of Aug 1 is 214, otherwise it is 213
	for i in range(len(years)):
		if i == 0: #first year
			if leapYearCheck(years[i]):
				start_day = (214-1) #starting on the leap year Julien day of Aug 1
				end_day = 366
			else:
				start_day = (213-1)
				end_day = 365
			#get the data for first year
			time_bounds = [start_day, end_day]
			tmin1 = subsetClimate(tmin_filenames[i], var[0], space_bounds, time_bounds)
			tmax1 = subsetClimate(tmax_filenames[i], var[1], space_bounds, time_bounds)
		else:
			start_day = 0 #starting on the leap year Julien day of Aug 1
			if leapYearCheck(years[i]):
				end_day = (214-1)
			else:
				end_day = (213-1)
			#get the data for second year
			time_bounds = [start_day, end_day]
			tmin2 = subsetClimate(tmin_filenames[i], var[0], space_bounds, time_bounds)
			tmax2 = subsetClimate(tmax_filenames[i], var[1], space_bounds, time_bounds)			

	#now concatenate the two years
	Tmin = np.vstack((tmin1, tmin2))
	Tmax = np.vstack((tmax1, tmax2))
	return(Tmin, Tmax) #returns the Tmin and Tmax datasets
	
	#all climate variable accounted for...
	
	#main
	#Tmin_in = tmin #here we could add a np.random.random()
	#Tmax_in = tmax 

if __name__ == '__main__': 
	
	Tmin, Tmax = getClimate(1999)
	#test year looks like it works!
	
	#out = mpb.runModel(Tmin, Tmax, year = y)
	
