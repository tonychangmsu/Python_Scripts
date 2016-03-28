#Title: MPB_phen_post_processing.py
#Author: Tony Chang
#Abstract: Post processing analysis of MPB output model
#Creation Date: 01/27/2016
#Modified Dates: 01/28/2016

#local data directory : G:\MPB\dev_comp_date
#code directory : E:\MPB_model\MPB_phenology\python_code

import numpy as np
from matplotlib import pyplot
import netCDF4 as nc
from osgeo import gdal
import os
import scipy.stats as stats

def get_wbp_mask(year = 1980, wbp_thr=0.421):
	'''
	input reference year and a percent thresh hold and
	returns boolean mask of PIAL climate suitable habitat
	'''
	#get the WBP mask
	wbp_maskfile = 'E:\\MPB_model\\WBP_masks\\WBP_probs_%s.tif'%year
	wbp_ds = gdal.Open(wbp_maskfile).ReadAsArray()	
	wbp_bin_mask = np.where(wbp_ds < wbp_thr, 0, 1)
	wbp_bool_mask = np.where(wbp_ds < wbp_thr, False, True)
	return(wbp_bool_mask)
	
def calc_undev_area(begin_year, end_year, dev_cutoff = 1461):
	'''
	input the year to start and end analysis, the cutoff number of days for development
	and a wbp_ref_year to calculate the percent of landscape where mpb remains undeveloped.
	This code assumes a 4 year cutoff (Although Logan states 3 years (personal comm 12/2016)
	Returns the percent area of WBP climate suitable habitat that has MPB undeveloped.
	'''
	percent_array = np.zeros(end_year-begin_year)
	wbp_bool_mask = get_wbp_mask() #use default values
	#note that there are a total of 40004 cells for wbp climate suitable habitat
	for year in range(begin_year,end_year):
		filename = 'stage_8_dev_comp_%s_test.nc' %(year)
		ds = nc.Dataset(filename)
		#get all the development_dates into a single array
		data = ds.variables['development_date'][:]
		masked_undev = np.where(data >= dev_cutoff, 1, 0)
		mk = np.sum(masked_undev, axis =0)
		bin_mask = np.where(mk >= 1, 1, 0)
		bool_mask = np.where(mk >= 1, True, False)
		full_mask = np.logical_xor(wbp_bool_mask, bool_mask)
		percent_undev = np.sum(bool_mk)/np.sum(wbp_bool_mask)
		percent_array[year-begin_year] = percent_undev
	return(percent_array)

def solve_adapt_emergence(year, dev_cutoff = 1461, wbp_apply=False):
	'''
	input the year perform analysis, the cutoff number of days for development
	and a wbp_apply boolean to calculate the percent of landscape where mpb remains undeveloped.
	This code assumes a 4 year cutoff (Although Logan states 3 years (personal comm 12/2016)
	Returns the mode of emergence per pixel on the landscape and the count of that emergence 
	day given that a MPB egg was layed each day for 365 beginning at the year requested
	'''
	filename = 'stage_8_dev_comp_%s_test.nc' %(year)
	ds = nc.Dataset(filename)
	#get all the development_dates into a single array
	data = ds.variables['development_date'][:]
	masked_undev = np.where(data >= dev_cutoff, 1, 0)
	mk = np.sum(masked_undev, axis =0)
	bin_mask = np.where(mk >= 1, 1, 0)
	bool_mask = np.where(mk >= 1, True, False)
	em_day, em_count = stats.mode(data.astype(float16), axis = 0) 
	#calculates the mode of the emergence for the year and the count for that mode
	#place a mask on these modes to remove undeveloped regions
	#if using wbp mask, get it and apply
	if wbp_apply:
		wbp_bool_mask = get_wbp_mask()
		full_mask = np.logical_not(np.logical_xor(wbp_bool_mask, bool_mask))
	else:
		full_mask = bool_mask
	ma_em_day = np.ma.masked_array(em_day[0], mask = full_mask)
	ma_em_count = np.ma.masked_array(em_count[0], mask = full_mask)
	return(ma_em_day, ma_em_count)

def adaptive_emergence_ts(begin_year, end_year, dev_cutoff = 1461, wbp_apply=False):
	'''
	Solves adaptive emergence for all the years specified and returns the array
	'''
	em_array = []
	counts_array = []
	for year in range(begin_year, end_year):
		ma_em_day, ma_em_count = solve_adapt_emergence(year, wbp_apply = wbp_apply)
	em_array.append(ma_em_day)
	counts_array.append(ma_em_count)
	return(em_array, counts_array)

def emergence_demo(year, writefile, x=150, y=150, dev_cutoff = 1461):
	filename = 'stage_8_dev_comp_%s_test.nc' %(year)
	ds = nc.Dataset(filename)
	#get all the development_dates into a single array
	data = ds.variables['development_date'][:]/1.
	#now we need histogram of the average development date for a single cell
	test_point = data[:][x][y]
	#find the mode of the test_point
	em_day, em_count = stats.mode(test_point) #julian day 197 or Jul 15th
	#we want to check that this particular day also has 15 days before and 15 days after that are close in frequency
	day_buffer = 14
	counts, day = np.histogram(test_point, bins = np.arange(last_day))
	mid_day = em_day.astype(int)[0]
	lower = mid_day - day_buffer #Jul 1
	upper = mid_day + day_buffer #Jul 29
	num_em = np.sum(counts[lower:upper]) #387 emergents within this month period
	mu_em = np.mean(day[lower:upper])
	f, ax = plt.subplots(figsize = (16,10), dpi = 300)
	ax.bar(day[lower:upper], counts[lower:upper], width = 1)
	ax.set_xlim(np.min(day[lower:upper]), np.max(day[lower:upper]))
	ax.set_title('Frequency distribution of emergence day for cell[%s][%s]'%(x,y), fontsize = 16)
	ax.set_xlabel('Emergence day (Julian day)', fontsize = 16)
	ax.set_ylabel('Count of emergences', fontsize = 16)
	ax.grid()
	ax.tick_params(labelsize=16)
	plt.savefig(writefile, dpi = 300,  bbox_inches='tight')
	return(print('%s written!' %(writefile)))

def plot_undev_timeseries(percent_array, begin_year, end_year, writefile):
	years = np.arange(begin_year, end_year)
	f, ax = plt.subplots(figsize = (20,6), dpi = 300)
	ax.plot(years, percent_array, lw = 2)
	ax.set_xlim(begin_year, end_year-1)
	ax.grid()
	ax.set_title('Percent of undeveloped MPB within GYE PIAL climate suitable habitat', fontsize = 16)
	ax.set_xlabel('Years', fontsize = 16)
	ax.set_ylabel('Percent undeveloped MPB within PIAL CSH', fontsize = 16)
	ax.tick_params(labelsize=16)
	plt.savefig(writefile, dpi = 300,  bbox_inches='tight')
	return(print('%s written!' %(writefile)))

def plot_dev_grid(em, year, writefile):
	xmin = -112.39583333837999
	xmax = -108.19583334006
	ymin = 42.279166659379996
	ymax = 46.195833324479999
	ae = [xmin, xmax, ymin, ymax]
	wbp_mask = get_wbp_mask()
	fig, ax = plt.subplots(figsize=(18, 12), dpi = 300)
	ax.imshow(np.logical_not(wbp_mask), cmap ='Reds_r', extent = ae)
	cax = ax.imshow(em, cmap ='viridis', extent = ae)
	ax.set_title(r'$Dendroctnous\ ponderosae$ mode of development within GYE beginning in: Year %s' %(year), fontsize =18)
	ax.tick_params(labelsize=16)
	ax.set_xlabel('Longitude (DD)', fontsize = 16)
	ax.set_ylabel('Latitude (DD)', fontsize = 16)
	cbar = fig.colorbar(cax)
	cbar.set_clim(vmin = 0,vmax= 1820)
	cbar.set_label(label='Days to adult development',size=16)
	cbar.ax.tick_params(labelsize=16)
	plt.savefig(writefile, dpi = 300,  bbox_inches='tight')
	return()
	
if __name__ == "__main__":
	os.chdir("G:\MPB\dev_comp_date") #go to data directory	
	begin_year = 1948
	end_year = 2011
	percent_array = calc_undev_area(begin_year, end_year)
	writefile = 'E:\\MPB_model\\MPB_phenology\\out\\undev_ts_20160128.png'
	plot_undev_timeseries(percent_array, begin_year, end_year, writefile)
	em_writefile = 'E:\\MPB_model\\MPB_phenology\\out\\emerg_demo_20160128.png'
	emergence_demo(2000, em_writefile)
	#next calculate the adaptive emergence day and counts
	#we will perform this for select years of interest for demonstration purposes
	em, count = solve_adapt_emergence(1980, wbp_apply = True) #mid
	em2, count2 = solve_adapt_emergence(1975, wbp_apply = True) #high
	em3, count3 = solve_adapt_emergence(2000, wbp_apply = True) #low
	em_writefile = 'E:\\MPB_model\\MPB_phenology\\out\\1980_em_grid_20160128.png'
	plot_dev_grid(em, 1980,em_writefile)
	em_writefile = 'E:\\MPB_model\\MPB_phenology\\out\\1975_em_grid_20160128.png'
	plot_dev_grid(em2, 1975,em_writefile)
	em_writefile = 'E:\\MPB_model\\MPB_phenology\\out\\2000_em_grid_20160128.png'
	plot_dev_grid(em3, 2000,em_writefile)
	
	count_writefile = 'E:\\MPB_model\\MPB_phenology\\out\\1980_count_grid_20160128.png'
	plot_dev_grid(count, 1980,count_writefile)
	count_writefile = 'E:\\MPB_model\\MPB_phenology\\out\\1975_count_grid_20160128.png'
	plot_dev_grid(count2, 1975,count_writefile)
	count_writefile = 'E:\\MPB_model\\MPB_phenology\\out\\2000_count_grid_20160128.png'
	plot_dev_grid(count3, 2000,count_writefile)

		'''
		#so two things could matter. The em_count and the num_em 
		#which gives us a value of how many emergences are happening on the particular peak day and 
		#how many are occurring within 2 weeks before and after
		#now solve the mode on axis 0
		em_days, em_counts = stats.mstats.mode(m_data, axis = 0) #going to want to time this...
		#mask em_days and em_counts
		#em_days gives us the mode of emergence for the cell
		#em_counts gives us the synchronicity of that emergence date
		'''
		em_days.mask = dev_mask
		em_counts.mask = dev_mask
		plt.imshow(em_days[0])
		plt.colorbar()
		plt.show()
		fig, ax = plt.subplots(figsize=(18, 12))
		plt.imshow(em_counts[0])
		plt.colorbar()
		'''
		
		#get the whitebark pine mask
		WBP_maskfile = 'E:\\MPB_model\\WBP_masks\\WBP_probs_%s.tif'%year
		wbp_ds = gdal.Open(WBP_maskfile).ReadAsArray()
		
		wbp_numeric_mask = np.where(wbp_ds < wbp_thr, 0, 1)
		#wbp_ds[wbp_ds < wbp_thr] = False
		#wbp_ds[wbp_ds >= wbp_thr] = True
		#wbp_numeric_mask = wbp_ds
		full_numeric_mask = wbp_numeric_mask + numeric_mask
		un_dev_cells = len(np.where(full_numeric_mask ==2)[0])
		wbp_count_cells = len(np.where(full_numeric_mask ==1)[0])
		non_wbp_count_cells = len(np.where(full_numeric_mask ==0)[0])
		vals = [un_dev_cells, wbp_count_cells, non_wbp_count_cells]
		count_array.append(vals)
		percent_un_dev = un_dev_cells/wbp_count_cells
		percent_array.append(percent_un_dev)
		#this can be for our time series of undeveloped locations relative to wbp area
		full_numeric_mask[full_numeric_mask==2] = 0
		full_mask = np.ma.make_mask(full_numeric_mask)
		#plt.imshow(full_mask)

		#try new mask
		em_days.mask = np.logical_not(full_mask)
		em_counts.mask = np.logical_not(full_mask)
		em_days_array.append(em_days)
		em_count_array.append(em_counts)
		'''
		fig, ax = plt.subplots(figsize=(18, 12))
		plt.imshow(em_days[0])
		plt.colorbar()
		plt.show()
		fig, ax = plt.subplots(figsize=(18, 12))
		plt.imshow(np.ma.masked_where(wbp_numeric_mask ==0, wbp_numeric_mask), cmap ='Reds_r')
		plt.imshow(em_counts[0], cmap ='viridis')
		plt.colorbar()	
		
		#now repeat this for every year at first
		'''
		
		