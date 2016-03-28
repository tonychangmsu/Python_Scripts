'''
Title: disturb_detect.py
Date: Created on April 17, 2015
Author: Tony Chang
Abstract: Uses the enchanced wetness difference index defined as the difference in tasseled wap wetness for 
two or more dates to detect red-attack stands of lodgepole pine. A decrease in TCWet over time (negative EWDI value) is the
best overall indicator of conifer mortality (Collins and Woodcock 1996) 
'''

#TC_wet = 0.2626*B1+0.2141*B2+0.0926*B3+0.0656*B4-0.7629*B5-0.5388*B7
#defined by Huang et al 2002

#use landsat images from 1999-2008 for path 38, row 29 to represent core of GYE WBP
#Late summer and early fall (ranging from Aug. 9 to Oct. 6) images were chosen to minimize the effects of snow, snowmelt,
#and understory green-up on TCWet. All images were geometrically registered to a base Landsat image (Sept. 15,
# 1999 ETM+) to within 0.5-pixel root mean squared error
#
#since fires were also a problem in detection of disturbance they were filtered out of the analysis
#Perimeters of wildfires in the study area were obtained from three sources: US Forest Service Region 1 Fire History
#Layer (Smail and Tanke 2008), Yellowstone National Park Fire History Layer (Spatial Analysis Center 2005), and the
#federal Landscape Fire and Resource Management Planning Tools Project (LANDFIRE) Rapid Refresh product, which
#is derived from MODIS satellite imagery (LANDFIRE 2007). These three fire history data sets were rasterized to
#30-m pixels and combined so that a fire recorded in any layer resulted in a fire in the combined raster. Fire perimeters
#were separately calculated for each individual year, 1999–2008. Cumulative fire history masks were created for
#each year of imagery (i.e., the 2005 fire mask included fires from 1999, 2000, 2001, 2002, 2003, 2004, and 2005,
#whereas the 2004 mask did not include 2005 fires).

'''
Table 1. Landsat imagery used in this study
Date Sensor Notes
Sept. 15, 1999 ETM+ Geometric base image
Sept. 20, 2001 ETM+ Radiometric base image
Sept. 23, 2002 ETM+
Oct. 4, 2003 TM
Oct. 6, 2004 TM
Sept. 7, 2005 TM
Aug. 9, 2006 TM
Sept. 21, 2007 ETM+ SLC off
Aug. 22, 2008 ETM+ SLC off
'''

#for now let's just test the transform for the wetness index

import numpy as np
import matplotlib.pyplot as plt
import gdal
import os
import pandas as pd
#parameters
def tcWetTransform(data):
	beta1 = 0.2626
	beta2 = 0.2141
	beta3 = 0.0926
	beta4 = 0.0656
	beta5 = -0.7629
	beta7 = -0.5388
	tcwet = data['B1']*beta1+data['B2']*beta2+data['B3']*beta3+data['B4']*beta4+data['B5']*beta5+data['B7']*beta7
	return(tcwet)

def getMeta(l_source):
	mtl_name =  [f for f in os.listdir(l_source) if f.endswith('MTL.txt')] #find metadata file
	fname = r'%s\%s' %(l_source, mtl_name[0])
	lines = iter(open(fname).readlines())
	group = []
	value = []
	for i in lines:
		line = i.strip().replace('"','').split('=') #cut and split each line
		if line[0] == 'END':
			break
		elif (line[0] =='BEGIN_GROUP ') or (line[0] =='END_GROUP ') or (line[0] =='GROUP '):
			continue
		else:
			group.append(line[0][:-1])
			value.append(line[1])
	metadata = dict(zip(group, value))
	return(metadata)

def LT5ToLT7(data, band):
	#transforms the landsat 5 tm DN data to match with landsat 7 tm+ DNs
	#parameters for transformation DN7 = (slope * DN5) + intercept
	slope = [0.943, 1.776, 1.538, 1.427, 0.984, np.nan, 1.304]
	intercept = [4.21, 2.58, 2.50, 4.80, 6.96, np.nan, 5.76]
	if isinstance(band, str): #check if it is a string
		band = int(''.join(x for x in band if x.isdigit())) #convert to integer
	return(data* slope[band-1] + intercept[band-1])

def DN5ToRad(qcal, metadata, band):	
	#this is for convert DN5 to radiance
	#input the band as a integer or string (i.e. 1 or 'B1')
	#converts the DN values to Radiance then to ToA Reflectance as according to 
	# http://landsathandbook.gsfc.nasa.gov/pdfs/L5TMLUTIEEE2003.pdf and 
	# http://landsathandbook.gsfc.nasa.gov/pdfs/Landsat_Calibration_Summary_RSE.pdf
	if isinstance(band, str): #check if it is a string
		band = int(''.join(x for x in band if x.isdigit())) #convert to integer
	lmax = float(metadata['RADIANCE_MAXIMUM_BAND_%s'%(band)])
	lmin = float(metadata['RADIANCE_MINIMUM_BAND_%s'%(band)])
	qcalmax = float(metadata['QUANTIZE_CAL_MAX_BAND_%s'%(band)])
	qcalmin = float(metadata['QUANTIZE_CAL_MIN_BAND_%s'%(band)])
	#qcal = DN_data['B%s'%(band)].astype(float)
	radiance = ((lmax - lmin)/(qcalmax - qcalmin))*(qcal-qcalmin)+lmin
	#radiance = ((lmax - lmin)/qcalmax) * qcal + lmin
	return(radiance)

def DN7ToRad(DN7, band):
	#transforms landsat 7 TM+ DN data to radiance via Chander et al. 2009
	# radiance = (gain * DN7) + bias
	gain = [0.778740,0.798819, 0.621654, 0.639764, 0.126220, np.nan 0.043898]
	bias = [-6.98, -7.20, -5.62, -5.74, -1.13, np.nan, -0.39]
	if isinstance(band, str): #check if it is a string
		band = int(''.join(x for x in band if x.isdigit())) #convert to integer
	return(data* gain[band-1] + bias[band-1])
	
def radToRef(L_lambda, metadata, band):
	if isinstance(band, str): #check if it is a string
		band = int(''.join(x for x in band if x.isdigit())) #convert to integer
	#ESUN = [1957,1826,1554,1036,215.0,np.nan, 80.67] #from http://landsathandbook.gsfc.nasa.gov/pdfs/L5TMLUTIEEE2003.pdf for NLAPS product
	ESUN = [1983,1796,1536,1031,220,np.nan, 83.44] #from http://landsathandbook.gsfc.nasa.gov/pdfs/Landsat_Calibration_Summary_RSE.pdf for LPGS product
	theta_s = float(metadata['SUN_ELEVATION'])
	esun_lambda = ESUN[(band-1)]
	d_data = pd.read_csv(r'K:\NASA_data\Landsat5\d.csv', skiprows=1, delimiter = ',')
	doy = int(metadata['LANDSAT_SCENE_ID'][12:17])
	d = d_data.d[doy+1]
	rho = (np.pi*L_lambda*(d**2))/(esun_lambda*cos(np.radians(theta_s)))
	return(rho)
	
def getLSdata(w_path, date_bounds, keybands = []):
	#now filter for the date that is desired
	#we will look for the September 15, 1999 file LT50380291999258 #don't have it
	#let's just try the first Summer image (between Aug 9 and Oct 6 (julien days 221-280)) adding one to account for leap years.
	if len(keybands) == 0:
		keybands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'] #if no input for the keybands, gather all bands
	scene_dirs = [x[0] for x in os.walk(w_path)]
	#loop through directories looking for the year and julien dates of interest
	fi_scenes = []
	for i in range(1,len(scene_dirs)): #skip the first directory
		ls_jul_date = int(scene_dirs[i][len(w_path)+14:len(w_path)+17])
		ls_year = int(scene_dirs[i][len(w_path)+10:len(w_path)+14])
		if (ls_year >= date_bounds['year'][0] and ls_year <= date_bounds['year'][1]):
			if (ls_jul_date >= date_bounds['julien_date'][0] and ls_jul_date <= date_bounds['julien_date'][1]):
				fi_scenes.append(i)
	#now we will just use the first fi_scene to perform our tc_wet calculation
	fi = 0
	l_source = scene_dirs[fi_scenes[fi]]
	#call get metadata function
	metadata = getMeta(l_source)
	f_names =  [f for f in os.listdir(l_source) if f.endswith('.TIF')] #find only the .TIF files
	DN_data = []
	ref_data = []
	for i in keybands:
		for j in f_names:
			if (i in j):
				#get tiff
				ds = gdal.Open(r'%s\%s'%(l_source, j)).ReadAsArray()
				#transform the ds of DN5 to DN7
				ds = LT5ToLT7(ds, i)
				DN_data.append(ds)
				#calculate the reflectance
				rad = DNToRad(ds, metadata, i)
				ref = radToRef(rad, metadata, i)
				ref[ref<0] = 0 # max sure all values are positive
				ref_data.append(ref)
				ds = None #close data
				continue
	d_dict = dict(zip(keybands, DN_data)) # create dict of the data
	r_dict = dict(zip(keybands, ref_data))
	return(r_dict, d_dict, metadata)


pi = 38
ri = 29

dir = r'p%sr%s'%(pi,ri)
wd = r'K:\NASA_data\Landsat5\%s'%(dir)
path_dirs = os.listdir(wd) 
order_name = [s for s in path_dirs if "Bulk Order" in s] #get the name of the bulk order directory 
w_path = r"%s\%s\l4-5 tm" %(wd, order_name[0])#assumes only one directory
date_bounds = {'year':[1999,2011],'julien_date':[221,280]}

ref_data, DN_data, metadata = getLSdata(w_path, date_bounds)
#wetness =  tcWetTransform(data)

#image processing stuff
#http://nbviewer.ipython.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/color_image_processing.ipynb
# may need to first transform the Landsat 5 TM DN data to Landsat 7 ETM+ DN data?
# images are still in DN so they need to be changed to top of atmosphere (ToA) reflectance
#http://www.yale.edu/ceo/Documentation/Landsat_DN_to_Reflectance.pdf
# need to gather the metadata
# now turn to radiance values
#meta data found  
# reflectance calculated
# calculated the tasseled cap transformation via #http://ibis.colostate.edu/webcontent/ws/coloradoview/tutorialsdownloads/co_rs_tutorial10.pdf

'''
img432 = np.dstack((data['B4'], data['B3'], data['B2']))
def color_image_show(img):
	"""
	Show image
	Input:
	img - 3D array of uint16 type
	title - string
	"""
	fig = plt.figure(figsize=(10, 10))
	fig.set_facecolor('white')
	plt.imshow(img/65535)
	plt.show()

plt.imshow(img432)
#print the histogram of the colors
fig = plt.figure(figsize=(10, 7))
fig.set_facecolor('white')

for color, channel in zip('rgb', np.rollaxis(img432, axis=-1)):
    counts, centers = exposure.histogram(channel)
    plt.plot(centers[1::], counts[1::], color=color)
plt.show()

#plot equalized image
img432_ha = np.empty(img432.shape, dtype='uint16')
lims = [(0,120),(0,120),(0,120)]
for lim, channel in zip(lims, range(3)):
    img432_ha[:, :, channel] = exposure.rescale_intensity(img432[:, :, channel], lim)

color_image_show(img432_ha)
'''



	





