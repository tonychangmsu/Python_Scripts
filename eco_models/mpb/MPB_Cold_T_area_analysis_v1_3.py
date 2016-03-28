#Title: MPB_Cold_T_gridtest.py
#Author: Tony Chang
#Date: 02.09.2015
#Abstract: Testing the functionality of the MPB_coldT_model.py functions under a multiple years of
#gridded climate data from 1948 to 2011?


import MPB_coldT_model_v2 as mpb
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import time as ts
import shapefile
import gdal
import geotool as gt

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
		else: #second year
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

def createOutNC(outdata, y, AOA, name = 'model_out'):
	#requires the AOA
	#function to generate a new NetCDF4 file to store output from the model
	n = len(outdata[0]) #number of days for the year
	nrows = np.shape(outdata[0][0])[0] #471
	ncols = np.shape(outdata[0][0])[1] #504
	dataset = nc.Dataset(name, 'w', format = 'NETCDF4_CLASSIC')
	#create all the dimensions
	time = dataset.createDimension('time', None)
	lat = dataset.createDimension('lat', nrows)
	lon = dataset.createDimension('lon', ncols)
	#create all the variables
	store_dtype = np.float32 #assign a different storage type to save memory
	time = dataset.createVariable('time',np.float64, ('time',))
	latitude = dataset.createVariable('latitude',np.float32, ('lat',))
	longitude = dataset.createVariable('longitude',np.float32, ('lon',))
	#output variables
	tau = dataset.createVariable('tau', store_dtype,('time','lat','lon',))
	R = dataset.createVariable('R', store_dtype,('time','lat','lon',))
	ptmin = dataset.createVariable('ptmin', store_dtype,('time','lat','lon',))
	lt50 = dataset.createVariable('lt50', store_dtype,('time','lat','lon',))
	survival = dataset.createVariable('survival', store_dtype,('time','lat','lon',))
	C = dataset.createVariable('C', store_dtype,('time','lat','lon',))
	P1 = dataset.createVariable('P1', store_dtype,('time','lat','lon',))
	P2 = dataset.createVariable('P2', store_dtype,('time','lat','lon',))
	P3 = dataset.createVariable('P3', store_dtype,('time','lat','lon',))
	#metadata regarding dataset
	dataset.description = 'Output from the MPB cold tolerance model'
	#dataset.history = 'Created %s' %(time.ctime(time.time()))
	dataset.source = 'MPB_Cold_T v2'
	dataset.Conventions = 'CF-1.6'
	dataset.institution = 'MSU Ecology Landscape Biodiversity Lab'
	dataset.title = 'Daily population metrics of mountain pine beetle as a function of temperature'
	dataset.comment = '30-arcsec spatial resolution, daily timestep'
	# Variable Attributes
	latitude.units = 'degrees north'
	longitude.units = 'degrees east'
	time.units = 'days since %s-08-01'%(y)
	survival.units = 'percent population survived'
	time.calendar = 'gregorian'
	#assign the values to each variable
	lats = np.linspace(AOA[1], AOA[3], nrows) #hard coded GYE bounds
	lons = np.linspace(AOA[0], AOA[2], ncols)
	latitude[:] = lats
	longitude[:] = lons
	time[:] = np.arange(0, n)
	#assign values to each output variable
	tau[:] = outdata[0][:] 	#this is all hard coded to work with the output from MPB_coldT_model_v2.py
	R[:] = outdata[1][:] 	#could be modified....
	ptmin[:] = outdata[2][:]
	lt50[:] = outdata[3][:]
	survival[:] = outdata[4][:]
	C[:] = outdata[5][:]
	P1[:] = outdata[6][:]
	P2[:] = outdata[7][:]
	P3[:] = outdata[8][:]
	dataset.close()
	return()

if __name__ == '__main__': 
	#topowx years range from 1948 to 2011
	start = 1948
	end = 2011
	years = np.arange(start, end)
	name = 'GYE_mpb_out_'	
	AOA = [-112.39166727055475, 42.27499982, -108.19166728736126, 46.19166648] #bounds for the GYE
	
	for y in years:
		Tmin, Tmax = getClimate(y)
	#test year looks like it works!
		if y == start:
			out = mpb.runModel(Tmin, Tmax, year = y)
		else:
			out = mpb.runModel(Tmin, Tmax, year = y, C_p = out[5][-1])
		createOutNC(out, y, AOA, name = 'K:\\NASA_data\\MPB_model_out\\%s%s.nc'%(name,y)) #save all the outputs
	
	
	#sname = 'D:\\CHANG\\GIS_Data\\GYE_Shapes\\GYE.shp' #reference file for the GYE shape
	#rname = 'E:\\TOPOWX\\GYE\\tmin\\TOPOWX_GYE_tmin_1_1948.tif' #reference file for climate data
	#mask, pnts, fex = shapeMask(sname, rname)
	
	