#Title: 	dev_completion.py
#Author:	Tony Chang
#Date: 		2016.01.16
#Abstract:	library to determine when the development will complete for an adult MPB 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import netCDF4 as nc
import scipy.integrate as sp
import timeit
from numba import double, jit
import gdal as gdal
#solve the cumulative integral for one year

def index_array(idx):
#inputs the indices for the end of development
#outputs 3 arrays for indexing the next year
	idc = np.indices(np.shape(idx))
	re_idx = np.reshape(idx, (np.shape(idx)[0]*np.shape(idx)[1]))
	re_idc_r = np.reshape(idc[0], (np.shape(idx)[0]*np.shape(idx)[1]))
	re_idc_c = np.reshape(idc[1], (np.shape(idx)[0]*np.shape(idx)[1]))
	return([re_idx, re_idc_r, re_idc_c])

def write_dev_comp_data(nc_ds, dev_comp_date, julian_start, year, outname, appnd = False):
	#inputs the topowx hourly data as a reference for the development rate write
	#outputs the development completion date into a netCDF4 file on disk
	if appnd: 
		root_grp = nc.Dataset(outname, 'a')
		#find length and add the data
		n = len(root_grp.variables['time'])
		root_grp.variables['time'][n] = julian_start
		root_grp.variables['development_date'][n,:,:] = dev_comp_date
		root_grp.close()
		return(print('%s updated with day %s' %(outname, julian_start)))

	else: #write a new file
		lat_array = nc_ds.variables['latitude']
		lon_array = nc_ds.variables['longitude']
		
		root_grp = nc.Dataset(outname, 'w', format = 'NETCDF4')
		root_grp.description = 'Days to development completion since start date'
		root_grp.history = 'Created %s' %(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
		root_grp.source = 'Montana State University Landscape Biodiversity Lab'

		# dimensions
		root_grp.createDimension('time', None) #infinite dimension
		root_grp.createDimension('lat', len(nc_ds.variables['latitude'][:]))
		root_grp.createDimension('lon', len(nc_ds.variables['longitude'][:]))

		# variables
		times = root_grp.createVariable('time', 'f4', ('time',))
		latitudes = root_grp.createVariable('latitude', 'f4', ('lat',))
		longitudes = root_grp.createVariable('longitude', 'f4', ('lon',))
		development_date = root_grp.createVariable('development_date', 'uint16', ('time', 'lat', 'lon',)) #unsigned 16 int
		
		# descriptions
		latitudes.units = 'degrees_north'
		longitudes.units = 'degrees_east'
		development_date.units = 'julian day'
		times.units = 'Start date of MPB cycle in Julian days since %s-%s 0:0:0'%(year, julian_start)
		times.calendar = 'Julian calendar'
		
		latitudes[:] = nc_ds.variables['latitude'][:]
		longitudes[:] = nc_ds.variables['longitude'][:]
		times[0] = julian_start #initial time
		development_date[0,:,:] = dev_comp_date #initial development_date
		root_grp.close()
		return(print("%s written!"%(outname)))

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
				
########################################################################	
######################### MAIN #########################################
########################################################################

stages = 8
nrows = 471
ncols = 504
startyear = 2000
endyear = 2015
nyears = 5 #maximum number of years for development before cutoff

last_stage = 8
for year in range(startyear,endyear):
	#perform this for every year.
	if leapYearCheck(year):
		ndays = 366
	else:
		ndays = 365


	#one option is to gather 4 years of data at a time.
	#then we can just import the files once in the beginning. We can try two years at a time....
	#two years at a time didn't work out well
	#the best solution could be to create one giant file for each stage. Then slice the time series to manageable bits

	#first get the data

	'''
	have a problem that even after 10 years, we can not get beetles to develop in the high elevation regions
	past the sixth stage..
	it is unlikely that the individual can survive beyond 2 years. This means that we could just make an assumption
	that the maximum number of years at 4 to be conservative.
	Bentz 2008 'Mountain Pine Beetle, Dendroctonus ponderosae (Coleoptera: Curculionidae, Scolytinae)'. Encyclopedia of Entomology.  pg.2494-2496 

	"The lifecycle of the mountain pine beetle is
	highly dependent upon temperature. Commonly,
	populations are univoltine, although at higher elevations
	where average temperatures are colder, 2
	and sometimes 3 years are required to complete a
	generation. Adult beetles emerge from host trees
	and disperse to new hosts in the warm summer
	months when temperatures are above 15.5°C.
	Although timing of emergence will vary from
	year to year depending on beetle development
	and temperature, peak adult emergence typically
	occurs within a 2–3 week time span. Rapid and
	synchronous emergence of the population is
	essential for mountain pine beetle success in overcoming
	the resinous defenses of healthy host trees"

	This could be an issue, because it is clear that beetles are attacking the high elevation zones
	'''


	#dev_comp is going to be the size of 1.7 gigs
	#we can't make an enormous array of that size unless we chunk it up.
	store_inx = np.zeros(last_stage, dtype='(%i,%i,%i)uint16' %(ndays,nrows,ncols))

	#indexing works by
	#store_inx[stage_number][julian_day][index_complete]
	#what we have is a single 1827x471x504 array that represents the d/dt rate of beetles for one stage
	#we will solve where each cell = 1 as the end of that stage
	#since, we will not progress through the next stage until the days are done, we will need to store 
	#each stage and day end date index within an ndays x 8 x 471 x 504 
	for stage in range(last_stage):
		for i in range(nyears):
			filename =  "K:\\NASA_data\\mpb_phen_out\\daily\\stage_%s\\mpb_phen_stage_%s_%s.nc"%(stage+1,stage+1,year+i)
			ds = nc.Dataset(filename)
			if i == 0:
				dev = ds.variables['development']
			else:
				dev = np.concatenate((dev, ds.variables['development']))
		#loop for the particular stage for the given start date
		#try for just 3 iterations
		dev_comp = np.cumsum(dev, axis = 0)
		#save this stage
		#find out where it equals 1 for all the cells
		#need to make nday number of dev_comp_off arrays for each day.
		#initialize the dev_comp_off with zero.
		#this is not too big but stores the indices of the last stage
		for julian_start in range(ndays): 
			#offset everything by the julian start date
			if julian_start == 0:
				dev_comp_off = dev_comp
			else:
				dev_comp_off = dev_comp - dev_comp[julian_start-1]
				
			#figure out if it is stage 1, then we always calculate from the first 1 encountered	
			if stage == 1:
				idx = np.argmax(dev_comp_off>=1, axis = 0)
				store_inx[stage][julian_start] = idx.astype(uint16)
				
			#if it is not stage 1, then we need to use the end date of the last stage to offset the dev_comp_off
			#even more from the last one
			else:
				#dev_prev_end = np.reshape(dev_comp_off[prev_stage_end], np.shape(idx))
				last_comp = index_array(store_inx[stage-1][julian_start])
				dev_prev_end = np.reshape(dev_comp_off[last_comp], np.shape(store_inx[stage-1][julian_start]))
				dev_comp_sub = dev_comp_off-dev_prev_end
				idx = np.argmax(dev_comp_sub>=1, axis = 0)
				idx[idx==0] = len(dev)-1 #does not complete stage for the year and julian start and set to last day...
				store_inx[stage][julian_start] = idx.astype(uint16)
				
			#prev_stage_end = index_array(idx)
			#prev_stage_end = index_array(store_inx[stage-1][julian_start])
			#dev_comp_dates[julian_start,stage-1] = idx
			#write at each julian day
			if stage == 7: #write out the last stage development
				#write the output
				#outname = 'K:\\NASA_data\\mpb_phen_out\\dev_comp\\stage_%s_dev_comp_test%s.nc'%(stage,year)
				outname = 'G:\\MPB\\dev_comp_date\\stage_%s_dev_comp_%s_test.nc'%(stage+1,year)
				if julian_start == 0:
					appnd = False
				else:
					appnd = True
				write_dev_comp_data(ds, store_inx[stage][julian_start], julian_start+1, year, outname, appnd = appnd)
	'''
	f, ax = plt.subplots(1,8, figsize = (16,12))
	for i in range(8):
		ax[i].imshow(dev_comp_dates[i], cmap = 'viridis')
		#ax[i].colorbar()
	'''
	#now we need to start a netCDF file to save the outputs for every day for the five year period
	#this would be the easiest way, so we don't overuse the memory in the computer to store the outputs.

		
	'''
	julian_start = 1

	#set a different limit...
	#subtract 365 from the indices, so the last year will not be counted

	sub_years = 1
	sub_days = (365 * sub_years) - julian_start
	z = dev_comp_dates[-1]
	undev = np.shape(np.where(z >= np.max(z)-sub_days))[1]
	z[z >= np.max(z)-sub_days] = np.nan
	max_dev_date = np.nanmax(z)

	import matplotlib.cm
	import matplotlib.patches as mpatches

	xmin = -112.39583333837999
	xmax = -108.19583334006
	ymin = 42.279166659379996
	ymax = 46.195833324479999
	ae = [xmin, xmax, ymin, ymax]

	fig, ax = plt.subplots(figsize=(20,16))
	ax.set_title(r'$Dendroctnous\ ponderosae$ development within GYE beginning in: Julian day %s, year %s' %(julian_start,year), fontsize =18)
	ax.tick_params(labelsize=16)
	ax.set_xlabel('Longitude (DD)', fontsize = 16)
	ax.set_ylabel('Latitude (DD)', fontsize = 16)
	cax = ax.imshow(z, cmap='viridis', extent = ae)
	cbar = fig.colorbar(cax)
	cbar.set_clim(vmin = 0,vmax= 1820)
	cbar.set_label(label='Days to adult development',size=16)
	cbar.ax.tick_params(labelsize=16)

	#make a masked array for the nan values of incomplete development
	masked_array = np.ma.array (z, mask=np.isnan(z))
	cmap = matplotlib.cm.viridis
	undev_color = 'white'
	cmap.set_bad(undev_color, alpha=0) #set nan values to transparent to see hatches
	ax.imshow(masked_array, cmap=cmap, extent = ae)

	#generate hatch if desired based on size of the image
	#xmin, xmax = ax.get_xlim() #unnecessary because we already explicitly defined the bounds
	#ymin, ymax = ax.get_ylim() #
	xy = (xmin,ymin)
	width = xmax - xmin
	height = ymax - ymin
	p = mpatches.Rectangle(xy, width, height, hatch='xxxxx', fill=None, zorder=-10)
	ax.add_patch(p)

	#generate legend for the hatches
	undev_patch = mpatches.Patch(color='black', label='Incomplete development', hatch ='xxxxx', fill =False)
	ax.legend(handles=[undev_patch], fontsize = 16)
	fig.savefig('E:\\MPB_model\\MPB_phenology\\out\\dev_%s.png'%year, transparent = True, bbox_inches='tight', dpi=300)

	Looking at the output for the final year, we can see that only 735 cells (adding one year
	made a change to 570 undeveloped cells) have not fully reached development.
	Which for the region represents only 0.3% of all the cells
	This could be substantially small if we only consider PIAL habitat...
	'''


	#need to make a check that if we subtract this index from the cumulative sum we get back to the original 
	#development per day rate, check the 0,0 element
	#r = 100
	#c = 100
	#end_date = idx[r][c]
	#checker = (dev_comp[:,r,c]-dev_comp[end_date,r,c])[end_date + 1]
	#b_out = ds.variables['development'][end_date + 1,r,c]
	#within 1e-8!
	'''

	#now we need to get the next stage
	stage = 2
	filename =  "K:\\NASA_data\\mpb_phen_out\\daily\\stage_%s\\mpb_phen_stage_%s_%s.nc"%(stage,stage,year)
	ds = nc.Dataset(filename)

	#solve for the cumulative sum, starting at a different date for each one
	#we can get the cumulative sum starting at the first day
	dev_comp_2 = np.cumsum(ds.variables['development'], axis = 0)

	#now subtract the development from the last day of the previous stage
	dev_prev_end = np.reshape(dev_comp_2[prev_stage_end], np.shape(idx))
	dev_comp_sub = dev_comp_2-dev_prev_end
	idx_next = np.argmax(dev_comp_sub>=1, axis = 0)

	#we have a problem when the np.argmax returns 0, because we don't get a >= 1 value.
	'''