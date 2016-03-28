"""
Title: Open/parse FIA data
Author: Tony Chang
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm 
import matplotlib.mlab as mlab
def plot_fia(data):
	plt.scatter(data['LON'],data['LAT'])
	return()

def data_AOA(AOA, plot):
	#returns the data filtered by bounding box AOA
	lat = plot['LAT']
	lon = plot['LON']
	lat_cut = (lat>=AOA[1]) & (lat<=AOA[3])
	lon_cut = (lon>=AOA[0]) & (lon<=AOA[2])
	return(plot[lat_cut & lon_cut])

def remove_dup(data):
	#sorts the unique CN values
	u, i = np.unique(data['CN'], return_index=True)
	return(data[i])
	
workspace = 'E:\\FIA\\'
name = 'fia_obs_clean4.10.2014.csv'
filename = '%s%s' %(workspace,name)

fia = np.genfromtxt(filename, delimiter = ',', names = True)

xmax = -108.263
xmin = -112.436
ymin = 42.252
ymax = 46.182
AOA = [xmin, ymin, xmax, ymax] #specify the bounds for the FIA data

prevalence = []
n = []
r = np.arange(1,10,0.1)
for i in r:
	DBHlimit = i# 1.57 #decide on the limit that define "sub-adult" 	Note: change from 3" to 2" only looses 17 trees...
	ht_limit = 12
	elev_limt = 4500 #5000, 5500, 6000, 6500, 7000 #whitebark pine limit seems to be at 7000
	wbp_code = 101
	limber_code = 113

	ffia = data_AOA(AOA, fia)
	wbp = ffia[(np.where(ffia['SPCD']==wbp_code)[0])]
	lbp = ffia[(np.where(ffia['SPCD']==limber_code)[0])]
	sap = wbp[(wbp['DIA'] < DBHlimit) & (wbp['HT'] <ht_limit)]
	lim_sap = lbp[(lbp['DIA'] < DBHlimit) & (lbp['HT'] <ht_limit) & (lbp['ELEV'] )]
	absent = ffia[(np.where(ffia['SPCD']!=wbp_code)[0])]

	uni_sap = remove_dup(sap)
	uni_lim_sap = remove_dup(lim_sap)
	uni_abs = remove_dup(absent)

	n_sap = len(uni_sap)
	n.append(n_sap)
	n_lbp = len(uni_lim_sap)
	n_abs = len(uni_abs)
	prevalence.append( (n_sap + n_lbp) / n_abs)


#===================================================#	
#==================PRE-ANALYSIS=====================#	
#===================================================#

#check number of saplings
print(len(uni_sap), len(uni_lim_sap))

plt.plot(r, n)
plt.plot(r, prevalence)

#consider the distribution of each species by elevation
# species : code
# pinus contorta : 108
# pseudotsuga menziesii : 202
# subalpine fir : 19
# whitebark pine = 101
# limber pine = 113
pcont_code = 108
pmenz_code = 202
sfir_code = 19

all_pcont = remove_dup(ffia[(np.where(ffia['SPCD']==pcont_code)[0])])
all_pmenz = remove_dup(ffia[(np.where(ffia['SPCD']==pmenz_code)[0])])
all_sfir = remove_dup(ffia[(np.where(ffia['SPCD']==sfir_code)[0])])
all_wbp = remove_dup(ffia[(np.where(ffia['SPCD']==wbp_code)[0])])
all_lbp = remove_dup(ffia[(np.where(ffia['SPCD']==limber_code)[0])])

#select the saplings
DBHlimit = 1.57
ht_limit = 12

pcont = all_pcont[(all_pcont['DIA'] < DBHlimit) & (all_pcont['HT'] <ht_limit)]
pmenz = all_pmenz[(all_pmenz['DIA'] < DBHlimit) & (all_pmenz['HT'] <ht_limit)]
sfir = all_sfir[(all_sfir['DIA'] < DBHlimit) & (all_sfir['HT'] <ht_limit)]
wbp = all_wbp[(all_wbp['DIA'] < DBHlimit) & (all_wbp['HT'] <ht_limit)]
lbp = all_lbp[(all_lbp['DIA'] < DBHlimit) & (all_lbp['HT'] <ht_limit)] #& (lbp['ELEV'] )]

#distributions seem normal so solve for the parameters...
#mu and sigma

pcont_para = norm.fit(pcont['ELEV'])
pmenz_para = norm.fit(pmenz['ELEV'])
sfir_para = norm.fit(sfir['ELEV'])
wbp_para = norm.fit(wbp['ELEV'])
lbp_para = norm.fit(lbp['ELEV'])

pcont_y = mlab.normpdf(pcont_dist[1], pcont_para[0], pcont_para[1])
pmenz_y = mlab.normpdf(pmenz_dist[1], pmenz_para[0], pmenz_para[1])
sfir_y = mlab.normpdf(sfir_dist[1], sfir_para[0], sfir_para[1])
wbp_y = mlab.normpdf(wbp_dist[1], wbp_para[0], wbp_para[1])
lbp_y = mlab.normpdf(lbp_dist[1], lbp_para[0], lbp_para[1])

#plot the different elevation histograms
plt.rcParams['figure.figsize'] = (18.0, 12.0)
b = 25
pmenz_dist = plt.hist(pmenz['ELEV'], bins = b, color = 'blue', alpha = 0.4, label = ' $\it pseudotsuga\,menziesii\,(n=%i,\mu =%.2f, \sigma=%.2f)$'%(len(pmenz),pmenz_para[0],pmenz_para[1]))
pcont_dist = plt.hist(pcont['ELEV'], bins = b, color = 'red', alpha = 0.4, label = '$\it pinus\,contorta\, (n=%i,\mu =%.2f, \sigma=%.2f)$'%(len(pcont),pcont_para[0],pcont_para[1]))
sfir_dist = plt.hist(sfir['ELEV'], bins = b, color = 'green', alpha = 0.4, label = ' $ \it abies\,lasiocarpa\, (n=%i,\mu =%.2f, \sigma=%.2f)$'%(len(sfir),sfir_para[0],sfir_para[1]))
wbp_dist = plt.hist(wbp['ELEV'], bins = b, color = 'yellow', alpha = 0.4, label = ' $ \it pinus\,albicaulis\, (n=%i,\mu =%.2f, \sigma=%.2f)$'%(len(wbp),wbp_para[0],wbp_para[1]))
lbp_dist = plt.hist(lbp['ELEV'], bins = b, color = 'black', alpha = 0.4, label = ' $ \it pinus\,flexilis\, (n=%i,\mu =%.2f, \sigma=%.2f)$'%(len(lbp),lbp_para[0],lbp_para[1]))
plt.grid()
plt.legend(loc = 'upper left')
plt.ylabel('Frequency')
plt.xlabel('Elevation')
#plot the lines
pmenz_wt = (pmenz_dist[1][1]-pmenz_dist[1][0])*(len(pmenz))
pcont_wt = (pcont_dist[1][1]-pcont_dist[1][0])*(len(pcont))
sfir_wt = (sfir_dist[1][1]-sfir_dist[1][0])*(len(sfir))
wbp_wt = (wbp_dist[1][1]-wbp_dist[1][0])*(len(wbp))
lbp_wt = (lbp_dist[1][1]-lbp_dist[1][0])*(len(lbp))

plt.plot(pmenz_dist[1], pmenz_y * pmenz_wt, color = 'blue', lw = 2)
plt.plot(pcont_dist[1], pcont_y * pcont_wt, color = 'red', lw = 2)
plt.plot(sfir_dist[1], sfir_y * sfir_wt, color = 'green', lw = 2)
plt.plot(wbp_dist[1], wbp_y * wbp_wt, color = 'yellow', lw = 2)
plt.plot(lbp_dist[1], lbp_y * lbp_wt, color = 'black', lw = 2)

plt.savefig('hist_sap.png', transparent=True, bbox_inches='tight', pad_inches=0)
#output is for the GYE range of the species
#=============================================================#
'''
What can we consider using these data histograms?
We can first see that the apparent (if we believe the data is unbiased)
lower bounds of whitebark pine is around 7000' in elevation. 
For limber pine, this is well within the mean of the sample population, therefore
it seems unfair to believe that any sapling that appears for limber pine above 7000' should
just be characterized as whitebark. However, if all the field measurements sub-adult measurements
below 7000' were classified as limberpine, it may be correct...

Perhaps we should just consider the saplings first?
'''