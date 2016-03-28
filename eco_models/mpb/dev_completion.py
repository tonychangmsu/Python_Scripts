#Title: 	dev_completion.py
#Author:	Tony Chang
#Date: 		2015.11.03
#Abstract:	library to determine when the development will complete for an adult MPB 

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import scipy.integrate as sp
import timeit
from numba import double, jit
import gdal as gdal
#solve the cumulative integral for one year


#first get the data
stage = 1
year = 1981
filename = "K:\\NASA_data\\mpb_phen_out\\stage_%s\\mpb_phen_stage_%s_%s.nc"%(stage,stage,year)
ds = nc.Dataset(filename)
#year2 = year+1
#filename2 = "K:\\NASA_data\\mpb_phen_out\\stage_%s\\mpb_phen_stage_%s_%s.nc"%(stage,stage,year2)
#ds2 = nc.Dataset(filename2)

#first thing may be to convert the units to days and divide by 1e6

#
d = ds.variables['development'][:]/1e4

#d = data[:]
#now find the cumulative integral

#t = ds.variables['time'][:]

t=ds.variables['time'][:]

#ones = np.ones(np.shape(d[0]))
#tm = np.tensordot(t, ones, axes=0)
#need time matrix, although the problem is that this takes too long given the memory needs

#trapz_out = sp.cumtrapz(d, tm, axis = 0, initial= 0)

#try to perform this iteratively? use the speed up with numba? for now lets just time 100x100
#remembering that all we really care about is the index of when the cumtrapz = 1 so first calculate the 
#cumtrapz for the particular cell, then save the index in a matrix

#shp = np.shape(d[0])
#holder = np.zeros(shp)

#apply a mask
mask_file = 'E:\\WBP_model\\output\\prob\\WBP2010_binary.tif'
mask = gdal.Open(mask_file).ReadAsArray()
bool_mask = np.where(mask==0, True, False)

#masked_d = np.ma.array(d, mask=d*mask[np.newaxis,:,:])
def solver(d, t):
	z,m,n = d.shape
	result = np.zeros((m,n)).astype(int)
	for i in range(m):
		for j in range(n):
			trapz_out = sp.cumtrapz(d[:,i,j]*24/1e6, t, initial =0)
			comp_index = np.argmax(trapz_out>1)
			result[i,j] = int(comp_index)
	return(result)
fast_solver = jit(double[:,:](double[:,:,:], double[:]))(solver)

tin = timeit.time.time()
out = fast_solver(ds.variables['development'][:],ds.variables['time'][:])
masked_out = np.ma.array(out, mask = bool_mask)
tout =  timeit.time.time()

print('Time = %s' %(tout-tin))
#this new method works, and only takes a minute. 

#perhaps limiting the analysis to only the places where Pinus albicaulis can grow is the best solution.
#problem with memory.
#need to devise a solution by chunking this at least into days.
#alternative solution is to compute the trapz for each set of days
#to do this we need to examine each of the time arrays
# and then determine the beginning and end of each day
#if we look at the floor of t, 
#basically we can get a floor of t array
t_floor = np.floor(t).astype(int)
#now set up a loop to solve the index for each whole integer or iterate up
#we can figure out if we have a leap year or not also by looking at the max day
#i.e. np.max(t_floor) == 365 , means this is a leap year dataset

max_days = np.max(t_floor)
z,m,n = d.shape
daily_dev = np.zeros((max_days, m,n))
start = 0 #first integration point
for k in range(max_days):
	ones = np.ones((m,n))
	end = np.argmax(t_floor>k) #find the end
	tm = np.tensordot(t[start:end], ones, axes=0)
	daily_dev[k] = np.trapz(d[start:end],tm,axis=0) 
	#define the new start
	print("start: %s, end: %s"%(start,end))
	start = end

	#now save as a daily development rate rather than hourly and use the cumulative sum method. 
	#we can save this as floating point, since the array size is small. 
	
#probably could perform this without the i and j loop, and just calculate np.trapz across axis 0 if t was a matrix of shape m, n