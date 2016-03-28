#Title: MPB_post_processing.py
#Author: Tony Chang
#Date: 02.10.2015
#Abstract: 	This script takes the output from MPB_Cold_T_area_analysis_v1_3.py that is stored as a NetCDF4
#			and compiles the data together to single variables, so that they can be accessed in a single manner
#			

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import shapefile
import gdal
import geotool as gt
import time 

def shapeMask(sname,rname):
	#takes a shapefile and generates a boolean mask from it (assumes single layer)
	raster_ref = gdal.Open(rname)
	sh_ds = shapefile.Reader(sname)
	shape = sh_ds.shapes()
	pnts = np.array(shape[0].points).T
	fextent = gt.getFeatureExtent(pnts)
	mask = gt.rasterizer(pnts, raster_ref)
	mask_bool = np.where(mask==0, True, False)
	return(mask_bool, pnts, fextent)
	
def outputAnnualSummary(v, startyear=1948, endyear =2011, mask = None):
	ws = 'K:\\NASA_data\\mpb_model_out\\Annual\\'
	f_head = 'GYE_mpb_out_'
	sdata_mu = [] #spatial data
	sdata_sd = []
	sdata_min = []
	sdata_max = []
	data_ts = []

	for y in range(startyear, endyear):
		filename = '%s%s%s.nc'%(ws, f_head, y)
		nc_ds = nc.Dataset(filename)
		#we want to get the annual average of the year and the standard deviations
		var_data = nc_ds.variables[v][:]
		smu = np.mean(var_data, axis = 0)
		ssd = np.std(var_data, axis = 0)
		smin = np.min(var_data, axis = 0)
		smax = np.max(var_data, axis = 0)
		if mask != None: #if there is a mask to be applied
			smu = np.ma.array(smu, mask = mask)
			ssd = np.ma.array(ssd, mask = mask)
			smin = np.ma.array(smin, mask = mask)
			smax = np.ma.array(smax, mask = mask)
		sdata_mu.append(smu) #get the mean for the area
		sdata_sd.append(ssd)
		sdata_min.append(smin)
		sdata_max.append(smax)

	#sdata_mu = np.array(sdata_mu)
	#sdata_sd = np.array(sdata_sd)
	if mask != None: 
		return(np.ma.array(sdata_mu), np.ma.array(sdata_sd), np.ma.array(sdata_min), np.ma.array(sdata_max))
	else:
		return(np.array(sdata_mu), np.array(sdata_sd), np.array(sdata_min), np.array(sdata_max))

def maskedMean(data):
	out = np.zeros(len(data))
	for i in range(len(data)):
		out[i] = np.ma.mean(data[i])
	return(out)
	
#var_values = ['tau', 'R', 'ptmin', 'lt50', 'survival', 'C', 'P1', 'P2' , 'P3' ]
startyear = 1948
endyear = 2011
'''
#if interested in the GYE average
sname = 'D:\\CHANG\\GIS_Data\\GYE_Shapes\\GYE.shp' #reference file for the GYE shape
rname = 'E:\\TOPOWX\\GYE\\tmin\\TOPOWX_GYE_tmin_1_1948.tif' #reference file for climate data
gyemask, pt, fex = shapeMask(sname, rname)
v_mu_gye, v_sd_gye = outputAnnualSummary(v_i, startyear, endyear, gyemask) 
'''

#could use the WBP 2010 mask (remembering to take the inverse because 1 is False and 0 is True)
wbpmask = ~np.array(gdal.Open('E:\\WBP_model\\output\\prob\\WBP2010_binary.tif').ReadAsArray(), dtype = bool)

v_mu_wbp, v_sd_wbp, v_min_wbp, v_max_wbp = outputAnnualSummary('survival', startyear, endyear, wbpmask) 
v_mu_tmin, v_sd_tmin, v_min_tmin, v_max_tmin = outputAnnualSummary('ptmin', startyear, endyear, wbpmask) 

#summarize the masked rasters
#ts_mu_gye = maskedMean(v_mu_gye)
#ts_sd_gye = maskedMean(v_sd_gye)
ts_mu_wbp = maskedMean(v_mu_wbp)
ts_sd_wbp = maskedMean(v_sd_wbp)
ts_min_wbp = maskedMean(v_min_wbp)
ts_max_wbp = maskedMean(v_max_wbp)
ts_mu_tmin = maskedMean(v_mu_tmin)
ts_sd_tmin = maskedMean(v_sd_tmin)
ts_min_tmin = maskedMean(v_min_tmin)
ts_max_tmin = maskedMean(v_max_tmin)

#set the outputs wanted

t = np.arange(startyear, endyear)

ts_m = ts_mu_wbp
ts_s = ts_sd_wbp

#sample from a truncated normal distribution at each timestep

from scipy.stats import truncnorm
from scipy.stats import norm

n = 200
rand_samp = np.zeros((n,len(t)))
u_lim = 1.0 #probabilities can run between 0 and 1
l_lim = 0.0
a = (l_lim - ts_m)/ts_s
b = (u_lim - ts_m)/ts_s
for i in range(n):
	for j in range(len(ts_m)):
		rand_samp[i][j] = truncnorm.rvs(a[j],b[j],loc=ts_m[j],scale=ts_s[j])

ax = plt.subplot(111)
for i in range(n):
	ax.plot(t, rand_samp[i], color = 'orange', alpha = 0.03)
ax.plot(t,ts_mu_wbp, color = 'red', lw = 1.5, label = 'wbp')
#ax2 = ax.twinx()
#ax2.plot(t,ts_mu_tmin, color = 'blue', lw = 1, ls = '--', label = 'wbp') 
#what you really want is the difference between minimum temp and lt50
ax.set_xlim(startyear, endyear)
ax.set_xlabel('$Time$', fontsize = 14)

#ax.tick_params(axis='y', colors='red')
ax.set_ylabel('$P(%s)$'%('survival'), fontsize = 14, color = 'red')
#ax2.tick_params(axis='y', colors='blue')
#ax2.set_ylabel(r'$\tau_{min}$ $(^oC)$', fontsize = 14, color = 'blue')
plt.title('GYE MPB population survival 1948-2011')
#plt.legend()
plt.grid()
plt.savefig('E:\\mpb_model\\climate_application\\output\\survival_plot_%s.png'%(time.strftime("%m%d%Y")
), dpi = 600, bbox_inches = 'tight')

#now we would like to highlight what the low point and the high points look like spatially
high = np.where(ts_mu_wbp ==np.max(ts_mu_wbp))[0][0]
low = np.where(ts_mu_wbp ==np.min(ts_mu_wbp))[0][0]
med = np.where(ts_mu_wbp ==np.median(ts_mu_wbp))[0][0]

plt.rcParams['figure.figsize'] = 10,8

fig = plt.figure()
ax1 = plt.subplot(131)
a = ax1.imshow(v_mu_wbp[low], vmin = 0, vmax = 1)
ax1 = plt.subplot(132)
b = ax1.imshow(v_mu_wbp[med], vmin = 0, vmax = 1)
ax2 = plt.subplot(133)
c = ax2.imshow(v_mu_wbp[high], vmin = 0, vmax = 1)
cbaxes = fig.add_axes([0.14,0.3, 0.76, 0.04]) 
cb = plt.colorbar(b, orientation ='horizontal', cax = cbaxes) 

