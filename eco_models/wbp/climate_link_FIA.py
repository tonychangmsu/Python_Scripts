import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import gdal as gdal

# Author: Tony Chang
# Abstract: Follows the FIA_preanalysis_08292014.py script. Output links FIA seedling points with PRISM climate outputs
# Date: 10/10/2014

##########################################################################################

def genLonLat(raster):
	#input gdal grid 
	#output lat/lon array 
	gt = raster.GetGeoTransform()
	xsize = raster.RasterXSize
	ysize = raster.RasterYSize
	lon = np.array([gt[0]+(i*gt[1]) for i in range(xsize)])
	lat = np.array([gt[3]+(i*gt[-1]) for i in range(ysize)])
	return(lon,lat)
	
def sampleIndexLink(data, raster): 
	#input the raw FIA data and a raw raster to link to
	#exports the indices that match sample spatial coordinates
	x,y = genLonLat(raster)
	n = len(data)
	sx = data.LON
	sy = data.LAT
	indexarray = {'sid':[],'xi':[],'yi':[]}
	for sample_i in range(n):
		ix = 0
		iy = 0
		indexarray['sid'].append(sample_i)
		if (sx[sample_i]>x.max() or sx[sample_i]<x.min() or sy[sample_i]>y.max() or sy[sample_i]<y.min()):#if the sample point is out of bounds, assign the index equal to nan
			ix = np.nan
			iy = np.nan
		else:
			while (sx[sample_i]>x[ix]): #since longitude is a negative value
				ix+=1
			while (sy[sample_i]<y[iy]): 
				iy+=1
		indexarray['xi'].append(ix)
		indexarray['yi'].append(iy)
	indexarray['sid'] = np.array(indexarray['sid'])
	indexarray['xi'] = np.array(indexarray['xi']) #reformat to numpy array
	indexarray['yi'] = np.array(indexarray['yi'])
	return(pd.DataFrame(indexarray))

def extractValues(ixs, raster):
	#input the indexarray from sampleIndexLink function and the raster from which values are to be extracted
	#outputs the point values to be extracted and linked
	values = []
	rasterarray = raster.ReadAsArray()
	for i in range(len(ixs)):
		if (np.isnan(ixs.xi[i]) or np.isnan(ixs.yi[i])):
			values.append(np.NaN)
		else:
			values.append(rasterarray[ixs.yi[i],ixs.xi[i]])
	return(np.array(values))

def linkData(data, raster, indiceslink, name):
	values = extractValues(indiceslink, raster)
	data[name] = pd.Series(values, index = data.index)
	return(data)

def covariate_filter(data, filterlist):
	#input the data array and variables desired by column name
	#outputs the culled data array including response and lat and long
	return(data[filterlist])
	
#open seedling dataset
type = 'SEEDMIX' #can run 'ADULT' or 'SEED' or 'SEEDMIX'
filename = 'E:\\FIA\\WBP_%s_PA.CSV' %(type)
data = pd.read_csv(filename)

#link to the dem to check for over deviation from coordinate fuzzing
dem = gdal.Open('E:\\GYE_TOPO\\dem_800m.tif')
indiceslink = sampleIndexLink(data, dem)
demValues = extractValues(indiceslink, dem)
data['dem'] = pd.Series(demValues*3.28084, index = data.index) #convert dem to ft

#link to all the climate variables from 1980-2010.
climateyear = 2010
variables = ['tmin', 'tmax', 'ppt', 'aet', 'pet', 'pack', 'soilm', 'vpd']
for v in variables:
	for m in range(1,13):
		raster = gdal.Open('E:\\PRISM\\30yearnormals\\%s\\%s_%s_%s_%s.tif'%(v, v, str(climateyear-30), str(climateyear), m))
		values = extractValues(indiceslink, raster)
		data['%s%s'%(v,m)] = pd.Series(values, index = data.index)

#filtering np.nan values
nan_mask = ~(np.isnan(data.dem))
data_cleaned = data[nan_mask]

#now filter out areas where the dem and elevation are different by 1000ft and remove np.nan values
dem_dev_limit = 984.252
fuzz_mask = np.abs(data_cleaned.ELEV-data_cleaned.dem)<dem_dev_limit
data_filtered = data_cleaned[fuzz_mask]

#####Some additional cleaning 
#rename the p_a column to match with the previous analysis label of 'response'
data_filtered.columns.values[1] = 'response' 
#write out this file for later use
data_filtered.to_csv('E:\\WBP_model\\New_Analysis\\FIA_%s_merged_cleaned.csv' %(type))
'''
####Generate the 2010 PRISM test data####
x, y = genLonLat(dem)
lonlat = np.meshgrid(x,y)
c = pd.DataFrame()
c['LAT'] = lonlat[1].reshape(lonlat[1].shape[0]*lonlat[1].shape[1])
c['LON'] = lonlat[0].reshape(lonlat[0].shape[0]*lonlat[0].shape[1])
#open the climate datasets
climatefoldername = 'E:\\PRISM\\30yearnormals\\'
climatelist = ['tmin', 'tmax', 'ppt', 'aet', 'pet', 'pack', 'soilm', 'vpd']
climateyear = 1980
for i in climatelist:
	for j in range(1,13):
		climate_data = gdal.Open('%s\\%s\\%s_%i_%i_%i.tif'%(climatefoldername, i, i, climateyear, climateyear+30, j)).ReadAsArray()
		c_reshaped = climate_data.reshape(climate_data.shape[0]*climate_data.shape[1])
		c['%s%s'%(i,j)] = pd.Series(c_reshaped)
c.to_csv('E:\\WBP_model\\fielddata\\PRISM_%s_%s_%s_data.csv'%(climateyear, climateyear+30))
'''