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
	sdata_med = []
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
		smed = np.median(var_data, axis = 0)
		ssd = np.std(var_data, axis = 0)
		smin = np.min(var_data, axis = 0)
		smax = np.max(var_data, axis = 0)
		if mask != None: #if there is a mask to be applied
			smu = np.ma.array(smu, mask = mask)
			smed = np.ma.array(smed, mask = mask)
			ssd = np.ma.array(ssd, mask = mask)
			smin = np.ma.array(smin, mask = mask)
			smax = np.ma.array(smax, mask = mask)
		sdata_mu.append(smu) #get the mean for the area
		sdata_med.append(smed)
		sdata_sd.append(ssd)
		sdata_min.append(smin)
		sdata_max.append(smax)

	#sdata_mu = np.array(sdata_mu)
	#sdata_sd = np.array(sdata_sd)
	if mask != None: 
		out = {'mu':np.ma.array(sdata_mu),'med':np.ma.array(sdata_med),'sd':np.ma.array(sdata_sd),'min':np.ma.array(sdata_min),'max':np.ma.array(sdata_max)}
	else:
		out = {'mu':np.array(sdata_mu),'med':np.array(sdata_med),'sd':np.array(sdata_sd), 'min':np.array(sdata_min), 'max':np.array(sdata_max)}
	return(out)

def maskedMean(data):
	out = np.zeros(len(data))
	for i in range(len(data)):
		out[i] = np.ma.mean(data[i])
	return(out)

def maskedPercentile(data, lower =5, upper=95):
	#takes in a masked array and returns the percentiles. Default is the 95 percentile
	low = np.zeros(len(data))
	high = np.zeros(len(data))
	for i in range(len(data)):
		low[i], high[i] = np.percentile(data[i].data[data[i].mask], (lower,upper))
	return(low, high)

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

v_surv_wbp = outputAnnualSummary('survival', startyear, endyear, wbpmask) 
v_tmin_wbp = outputAnnualSummary('ptmin', startyear, endyear, wbpmask) 
v_lt50_wbp = outputAnnualSummary('lt50', startyear, endyear, wbpmask) 

#summarize the masked rasters
#ts_mu_gye = maskedMean(v_mu_gye)
#ts_sd_gye = maskedMean(v_sd_gye)
ts_mu_wbp = maskedMean(v_surv_wbp['mu'])
ts_5_wbp, ts_95_wbp = maskedPercentile(v_surv_wbp['mu'])
ts_mu_tmin = maskedMean(v_tmin_wbp['mu'])
ts_min_tmin = maskedMean(v_tmin_wbp['min'])
ts_min_lt50 = maskedMean(v_lt50_wbp['min'])
ts_diff = ts_min_tmin-ts_min_lt50
#set the outputs wanted

t = np.arange(startyear, endyear)

#sample from a truncated normal distribution at each timestep
'''
"""
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
"""
high = np.where(ts_mu_wbp ==np.max(ts_mu_wbp))[0][0]
low = np.where(ts_mu_wbp ==np.min(ts_mu_wbp))[0][0]
med = np.where(ts_mu_wbp ==np.median(ts_mu_wbp))[0][0]

pnts = np.array([[t[low], t[med], t[high]],[ts_mu_wbp[low],ts_mu_wbp[med],ts_mu_wbp[high]]])
pntlabels = ['Low','Med','High']
txtpnts = np.array([[pnts[0][0],pnts[0][1],pnts[0][2]],[pnts[1][0]-0.06,pnts[1][1]-0.08,pnts[1][2]+0.02]])

plt.rcParams['figure.figsize'] = 8,4
ax = plt.subplot(111)
#ax.scatter(pnts[0], pnts[1], color ='blue')
#not considering the bootstrap sampling
#for i in range(n):
#	ax.plot(t, rand_samp[i], color = 'orange', alpha = 0.03)

ax.plot(t,ts_mu_wbp, color = 'red', lw = 1.5, label = 'wbp')
ax.fill_between(t, ts_5_wbp, ts_95_wbp, color = 'orange', alpha=0.2)
#ax.scatter(pnts[0], pnts[1], color ='blue')
#this plots temperature on the same figure
#ax2 = ax.twinx()
#ax2.plot(t,ts_mu_tmin, color = 'blue', lw = 1, ls = '--', label = 'wbp') 
#ax.tick_params(axis='y', colors='black')
#ax2.tick_params(axis='y', colors='blue')
#ax2.set_ylabel(r'$\tau_{min}$ $(^oC)$', fontsize = 14, color = 'blue')
#for i, txt in enumerate(pntlabels):
#	ax.annotate('%s'%(txt), (pnts[0][i], pnts[1][i]),xytext=(txtpnts[0][i],txtpnts[1][i]))
ax.set_ylabel('$P(%s)$'%('survival'), fontsize = 14, color = 'black')
ax.set_xlim(startyear, endyear)
ax.set_xlabel('$Time$', fontsize = 14)
plt.title('GYE MPB population survival 1948-2011')
plt.grid()
plt.savefig('E:\\mpb_model\\climate_application\\output\\survival_plot_%s.png'%(time.strftime("%m%d%Y")
), dpi = 600, bbox_inches = 'tight')
'''
#now we would like to highlight what the low point and the high points look like spatially
'''
plt.rcParams['figure.figsize'] = 10,8

ae = [-112.39166727055475, -108.19166728736126, 42.27499982, 46.19166648]
i=0
fig = plt.figure()
ax1 = plt.subplot(131)
a = ax1.imshow(v_surv_wbp['mu'][low], vmin = 0, vmax = 1, extent = ae)
ax1.set_title('%s (year = %i)' %(pntlabels[i],pnts[0][i]), fontsize = 16)
ax1.locator_params(nbins=4)
ax1.grid()
ax1.set_xlabel('Longitude (DD)')
ax1.set_ylabel('Latitude (DD)')

i =1
ax2 = plt.subplot(132)
b = ax2.imshow(v_surv_wbp['mu'][med], vmin = 0, vmax = 1, extent = ae)
ax2.set_title('%s (year = %i)' %(pntlabels[i],pnts[0][i]), fontsize = 16)
ax2.locator_params(nbins=4)
ax2.grid()
ax2.set_xlabel('Longitude (DD)')
ax2.set_ylabel('Latitude (DD)')

i=2
ax3 = plt.subplot(133)
c = ax3.imshow(v_surv_wbp['mu'][high], vmin = 0, vmax = 1, extent = ae)
ax3.set_title('%s (year = %i)' %(pntlabels[i],pnts[0][i]), fontsize = 16)
ax3.locator_params(nbins=4)
ax3.grid()
ax3.set_xlabel('Longitude (DD)')
ax3.set_ylabel('Latitude (DD)')

cbaxes = fig.add_axes([0.14, 0.2, 0.76, 0.04]) 
cb = fig.colorbar(b, orientation ='horizontal', cax = cbaxes) 
cb.set_label('$P(survival)$', fontsize = 18)
cb.ax.tick_params(labelsize = 14)
fig.tight_layout()
plt.savefig('E:\\mpb_model\\climate_application\\output\\survival_spatial_%s.png'%(time.strftime("%m%d%Y")
), dpi = 600, bbox_inches = 'tight')
'''
#now we need to plot the tmin and lt50 against time
'''
plt.rcParams['figure.figsize'] = 8,4
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(t, ts_min_tmin, color = 'blue', lw = 1.5, label = r'$\tau_{min}$')
ax.plot(t, ts_min_lt50, ls = ':', color = 'green', lw = 1.5, label = r'$LT_{50}$')
#ax.fill_between(t,ts_min_tmin,ts_min_lt50, color ='grey', alpha =0.2)
ax.grid()
ax.set_xlim(t[0], t[-1])
ax.set_xlabel('$Time$', fontsize = 18)
ax.set_ylabel('$Temperature$ $(^oC)$', fontsize =18)
ax.legend(loc='lower right')
plt.savefig('E:\\mpb_model\\climate_application\\output\\temp_compare_%s.png'%(time.strftime("%m%d%Y")
), dpi = 600, bbox_inches = 'tight')
'''
'''
plt.rcParams['figure.figsize'] = 12,4
import matplotlib.gridspec as gs
fig = plt.figure()
gax = gs.GridSpec(1,4) 
ax1 = plt.subplot(gax[0,:-1])
n_diff = (ts_diff-np.min(ts_diff))/(np.max(ts_diff)-np.min(ts_diff))
n_surv = (ts_mu_wbp-np.min(ts_mu_wbp))/(np.max(ts_mu_wbp)-np.min(ts_mu_wbp))
ax1.plot(t, n_diff, color = 'purple', ls="-.", lw = 2, label = r'$\widehat{\tau_{min}-LT_{50}}$')
ax1.plot(t, n_surv, color='red', label = r'$\widehat{P(survival)}$')
ax1.set_xlabel('$Time$', fontsize = 18)
ax1.set_ylabel('$Normalized$ $value$', fontsize = 18)
ax1.set_xlim(t[0], t[-1])
ax1.legend(loc ='lower right', fontsize = 10)
ax1.grid()

ax2 = plt.subplot(gax[0,-1])
ax2.scatter(n_diff, n_surv, marker = 'o')
ax2.plot(np.array([0,1]),np.array([0,1]), ls = ':', lw = 2)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel(r'$\widehat{\tau_{min}-LT_{50}}$', fontsize = 18)
ax2.set_ylabel(r'$\widehat{P(survival)}$', fontsize = 18)
ax2.grid()
fig.tight_layout()
plt.savefig('E:\\mpb_model\\climate_application\\output\\%s\\temp_thr_surv_%s.png'%(time.strftime("%m%d%Y"),time.strftime("%m%d%Y")
), dpi = 600, bbox_inches = 'tight')
'''
#plot with np.linalg.norm
'''
ax.grid()
ax.set_xlim(t[0], t[-1])
ax.legend()

ax2 = plt.subplot(212)
ax2.plot(t, ts_diff/np.linalg.norm(ts_diff), color = 'r')
ax2.plot(t, ts_mu_wbp/np.linalg.norm(ts_mu_wbp), color='blue')
plt.show()
'''