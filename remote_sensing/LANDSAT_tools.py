'''
Title: 		LANDSAT_tools.py
Date: 		Created 2015.04.17
			Modified 2016.01.21
Author: 	Tony Chang
Abstract: 	Uses the enchanced wetness difference index defined as the difference in tasseled wap wetness for 
two or more dates to detect red-attack stands of lodgepole pine. A decrease in TCWet over time (negative EWDI value) is the
best overall indicator of conifer mortality (Collins and Woodcock 1996) 
'''

#TC_wet = 0.2626*B1+0.2141*B2+0.0926*B3+0.0656*B4-0.7629*B5-0.5388*B7
#defined by Huang et al 2002

#use landsat images from 1997-2011 for path 38, row 29 to represent core of GYE WBP
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

#for now let's just test the transform for the wetness index
import sys
sys.path.append(r'K:\NASA_data\scripts') #add directory for geotools_LT script
import numpy as np
import matplotlib.pyplot as plt
import osgeo.gdal as gdal
import os
import pandas as pd
import shapefile
import geotool_LT as gt
from bs4 import BeautifulSoup as bs
#parameters

def get_metadata(l_source):
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

def get_xml_metadata(l_source):
	'''
	Reads the xml file for parameters and imports as a BeautifulSoup object
	'''
	xml_name =  [f for f in os.listdir(l_source) if f.endswith('.xml')] #find metadata file
	fname = r'%s\%s' %(l_source, xml_name[0])
	out = bs(open(fname).read(), 'lxml')
	return(out)

def get_solar_attributes(l_source):
	xml = get_xml_metadata(l_source)
	azi = float(xml.solar_angles['azimuth'])
	zen = float(xml.solar_angles['zenith'])
	ori = float(xml.orientation_angle.contents[0])
	return(ori, azi, zen)
	

def get_data_list(wd, year = None):
	''' 
	subset the dir_list for specified years if by_year is True
	similar to mod_file_search in MODIS_acquire.py library
	'''
	
	lt_dir_list = []
	lt_tif_list = []
	if year != None :
		sub_dir = [s for s in os.listdir(wd) if (('LT' in s) and (not '.' in s) and (int(s[9:13]) == year))] #exclude the .gz files
	else:
		sub_dir = [s for s in os.listdir(wd) if (('LT' in s) and (not '.' in s))] #exclude the .gz files
	for sub in sub_dir:
		fpath = "%s\\%s"%(wd,sub)
		lt_dir_list.append(fpath)
		for lt_file in os.listdir(fpath):
			if lt_file.lower().endswith('.tif'):
				lt_tif_list.append("%s\\%s"%(fpath, lt_file))
	return(lt_dir_list, lt_tif_list)

def lt_acquire_by_band(fpath, band_name):
	#input directory name and band name 
	#returns the individual band as a gdal Dataset
	lt_tif_list = []
	if str(band_name).isdigit():
		band_name = '%s'%(band_name)
	for lt_file in os.listdir(fpath):
		if lt_file.lower().endswith('%s.tif'%band_name):
			fname = '%s\\%s'%(fpath, lt_file)
			return(gdal.Open(fname))

def get_bands(fpath, data_type = None):
	#returns the individual 1-7 bands from Landsat  
	#in gdal dataset format
	lt_bands = []
	for i in range(1,8):
			if data_type != None:
				band = '%s_band%s' %(data_type, i)
			else:
				band = i
			lt_bands.append(lt_acquire_by_band(fpath, band))
	return(lt_bands)

def check_left_right(ds1,ds2):
	'''checks which dataset further 'left' or 'right' of the other'''
	transform1 = ds1.GetGeoTransform()
	transform2 = ds2.GetGeoTransform()
	if transform1[0] > transform2[0]:
		return(ds2, ds1)
	else:
		return(ds1, ds2)

def mosaic(ds1, ds2, stats_out = False, datafill = True, method = None):
	'''Assumes two scenes represent the same time of allocation'''
	#we would like ds1 to be on the left and ds2 to be on the right...
	ds1, ds2 = check_left_right(ds1,ds2)
	#takes in 2 gdal datasets to mosaic together
	band1 = ds1.GetRasterBand(1)
	rows1 = ds1.RasterYSize
	cols1 = ds1.RasterXSize
	datatype = ds1.GetRasterBand(1).DataType
	# get the corner coordinates for doq1
	transform1 = ds1.GetGeoTransform()
	minX1 = transform1[0]
	maxY1 = transform1[3]
	pixelWidth1 = transform1[1]
	pixelHeight1 = transform1[5]
	maxX1 = minX1 + (cols1 * pixelWidth1)
	minY1 = maxY1 + (rows1 * pixelHeight1)
	# read in doq2 and get info about it
	band2 = ds2.GetRasterBand(1)
	rows2 = ds2.RasterYSize
	cols2 = ds2.RasterXSize
	# get the corner coordinates for doq2
	transform2 = ds2.GetGeoTransform()
	minX2 = transform2[0]
	maxY2 = transform2[3]
	pixelWidth2 = transform2[1]
	pixelHeight2 = transform2[5]
	maxX2 = minX2 + (cols2 * pixelWidth2)
	minY2 = maxY2 + (rows2 * pixelHeight2)
	# get the corner coordinates for the output
	minX = min(minX1, minX2)
	maxX = max(maxX1, maxX2)
	minY = min(minY1, minY2)
	maxY = max(maxY1, maxY2)
	# get the number of rows and columns for the output
	cols = int(np.ceil((maxX - minX) / pixelWidth1))
	rows = int(np.ceil(((maxY - minY) / abs(pixelHeight1))))
	# compute the origin (upper left) offset for doq1
	xOffset1 = int((minX1 - minX) / pixelWidth1)
	yOffset1 = int((maxY1 - maxY) / pixelHeight1)
	# compute the origin (upper left) offset for doq2
	xOffset2 = int((minX2 - minX) / pixelWidth1)
	yOffset2 = int((maxY2 - maxY) / pixelHeight1)
	#we need to figure out which one is on the right and which one is on the left...
	# create the output image
	mem_drv = gdal.GetDriverByName('MEM')
	dsOut = mem_drv.Create('', cols, rows, 1, datatype)
	bandOut = dsOut.GetRasterBand(1) #empty for now
	# read in doq1 and write it to the output
	data1 = band1.ReadAsArray(0, 0, cols1, rows1)
	data2 = band2.ReadAsArray(0, 0, cols2, rows2)
	#one option is to combine it here...
	bandOut.WriteArray(data1, xOffset1, yOffset1)
	# read in doq2 and write it to the output
	bandOut.WriteArray(data2, xOffset2, yOffset2)
	if datafill == True:
	#modify data2 here to adjust for 0 values that can be filled by data1
		data1_fill = data1[:rows1-(rows-rows2),xOffset2:]
		data2_fill = data2[rows-rows1:, :cols1-xOffset2]
		if method == 'mean':
			data_fill = np.where(data2_fill == 0, data2_fill + data1_fill, (data2_fill + data1_fill)/2) 
		elif method == None:
			#we can select the scene with greater reflectance
			if np.mean(data1_fill) > np.mean(data2_fill):
				data_fill = np.where(data1_fill == 0, data2_fill + data1_fill, data1_fill) #assuming data1_fill is correct
			else:
				data_fill = np.where(data2_fill == 0, data2_fill + data1_fill, data2_fill) #assuming data1_fill is correct
		#could also average the two tiles
		#write in the datafill
		bandOut.WriteArray(data_fill, xOffset2, yOffset1)
	# compute statistics for the output
	bandOut.FlushCache()
	stats = bandOut.GetStatistics(0, 1)
	# set the geotransform and projection on the output
	geotransform = [minX, pixelWidth1, 0, maxY, 0, pixelHeight1]
	dsOut.SetGeoTransform(geotransform)
	dsOut.SetProjection(ds1.GetProjection())
	if stats_out:
		return(dsOut, stats)
	else:
		return(dsOut)

def multi_scene_mosaic():
	return()

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

def DN7ToRad(data, band):
	#transforms landsat 7 TM+ DN data to radiance via Chander et al. 2009
	# radiance = (gain * DN7) + bias
	gain = [0.778740,0.798819, 0.621654, 0.639764, 0.126220, np.nan, 0.043898]
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
	d_data = pd.read_csv(r'G:\NASA_remote_data\Landsat5\d.csv', skiprows=1, delimiter = ',')
	doy = int(metadata['LANDSAT_SCENE_ID'][12:17])
	d = d_data.d[doy+1]
	rho = (np.pi*L_lambda*(d**2))/(esun_lambda*np.cos(np.radians(theta_s)))
	return(rho)
	
def getLSdata(w_path, date_bounds, keybands = []):#, mask = []):
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
				#check if the user input a mask
				ds = LT5ToLT7(ds, i)
				#if len(mask) != 0:
				#	ds = np.ma.array(ds, mask = mask)
				DN_data.append(ds)	
				#calculate the reflectance
				#rad = DN5ToRad(ds, metadata, i) #changed to use the DN7 transform
				rad = DN7ToRad(ds, i)
				ref = radToRef(rad, metadata, i) #this reflectance algorithm only works for DN7 not DN5
				ref[ref<0] = 0 # max sure all values are positive
				ref_data.append(ref)
				ds = None #close data
				continue
	d_dict = dict(zip(keybands, DN_data)) # create dict of the data
	r_dict = dict(zip(keybands, ref_data))
	return(r_dict, d_dict, metadata)

def applyTransform(data, params):
	t = data['B1']*params[0]+data['B2']*params[1]+data['B3']*params[2]+data['B4']*params[3]+data['B5']*params[4]+data['B7']*params[6]
	return(t)
	
def TCTransform(data):
	#input the full reflectance dataset and calculates the brightness, greeness, and wetness indices via Huang et al 2002
	#parameters
	bright_param = [0.3561, 0.3972, 0.3904, 0.6966, 0.2286, np.nan, 0.1596]
	green_param = [-0.3344, -0.3544, -0.4556, 0.6966, -0.0242, np.nan, -0.2630]
	wet_param = [0.2626, 0.2141, 0.0926, 0.0656, -0.7629, np.nan, -0.5388]
	bright = applyTransform(data, bright_param)
	green = applyTransform(data, green_param)
	wet = applyTransform(data, wet_param)
	return(bright, green, wet)

def ndviTransform(ref_data):
	ndvi = (ref_data['B4']-ref_data['B3'])/(ref_data['B4']+ref_data['B3'])
	return(ndvi)
	
def LTshapeMask(sname,rname, pi, ri):
	#takes a shapefile and generates a boolean mask from it 
	#for LANDSAT data, given the path and row
	raster_ref = gdal.Open(rname)
	sh_ds = shapefile.Reader(sname)
	fields = sh_ds.fields
	#find where 'PATH' and 'ROW' are indexed
	for i in range(len(fields)):
		if fields[i][0] == 'PATH':
			p_ind = i-1
		if fields[i][0] == 'ROW':
			r_ind = i-1
	for rec in enumerate(sh_ds.records()):
		if ((rec[1][p_ind] == pi) and (rec[1][r_ind] == ri)):
			selection = rec[0] #index of the desired path and row shape
	shape = sh_ds.shape(selection)
	pnts = np.array(shape.points).T
	fextent = gt.getFeatureExtent(pnts)
	mask = gt.rasterizer(pnts, raster_ref)
	mask_bool = np.where(mask==0, True, False)
	return(mask_bool, pnts, fextent)

def getRasterExtent(metadata):
	xmin = float(metadata['CORNER_LL_PROJECTION_X_PRODUCT'])
	xmax = float(metadata['CORNER_UR_PROJECTION_X_PRODUCT'])
	ymin = float(metadata['CORNER_LL_PROJECTION_Y_PRODUCT'])
	ymax = float(metadata['CORNER_UR_PROJECTION_Y_PRODUCT'])
	return([xmin, xmax, ymin, ymax])

	#'CORNER_LL_LON_PRODUCT', 'CORNER_UR_LON_PRODUCT'
	#get the dataset now
	ref_data, DN_data, metadata = getLSdata(w_path, date_bounds)#, mask = mask_bool)
	brt, grn, wet = TCTransform(ref_data)
	ndvi = ndviTransform(ref_data)
	rfilename = metadata['FILE_NAME_BAND_1']
	rname = r'%s\%s\%s'%(w_path, rfilename[1:-7],rfilename[1:])
	sname = r'G:\NASA_remote_data\Landsat5\wrs2_descending\wrs2_descending_UTM_12N.shp'
	#code to select by attribute and get the points
	mask_bool, pnts, fextent = LTshapeMask(sname, rname, pi, ri)
	b3 = np.ma.array(ref_data['B3'], mask = mask_bool)
	b2 = np.ma.array(ref_data['B2'], mask = mask_bool)
	b1 = np.ma.array(ref_data['B1'], mask = mask_bool)
	img321 = np.dstack((b3,b2,b1)) #natural color
	
	#get the points for polygon
	yell_shpfile = r'G:\NASA_remote_data\shapes\yell_UTM12.shp'
	shp = shapefile.Reader(yell_shpfile).shape()
	rextent = getRasterExtent(metadata)
	pnts = np.array(shp.points)
	plt.plot(pnts[:,0], pnts[:,1], color = 'red', alpha = 0.5)
	plt.imshow(img321, extent = rextent)		
	plt.xlabel('E (m)')
	plt.ylabel('UTM 12 N (m)')
	#outdir = r'D:\CHANG\PhD_Material\Conferences_workshops\052015_NCCSC_OpenScience'
	outdir = r'K\\NASA_data\\script_out'
	#save the true color image
	plt.savefig(r"%s\%s"%(outdir,'true_color.png'), bbox_inches='tight', dpi =600)
	
	fig = plt.subplot(111)
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	plt.imshow(np.ma.array(grn, mask =mask_bool), cmap = 'Greens')
	plt.colorbar()
	plt.savefig(r"%s\%s"%(outdir,'greenness.png'), bbox_inches='tight', dpi =600)
