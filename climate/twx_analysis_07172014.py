'''
Created on Jul 17, 2014
Utility methods for calculating statistical summaries of TopoWx data
@author: tony.chang
'''


import numpy as np
from util_dates import YEAR,MONTH
import util_dates as utld
from scipy import stats
import osgeo.gdal as gdal
import osgeo.gdalconst as gdalconst
import osgeo.osr as osr
from netCDF4 import Dataset, num2date
import twx_sumry_v2 as twxs

class TairMosaic():
	'''
	A class for containing the full mosaic of arrays
	'''
	def __init__(self, 
if __name__ == '__main__': 

	#Example code for calculating annual trends for a TopoWx tile and outputting them as a GeoTiff
	workspace = 'E:\\TOPOWX\\h06v02\\topowx_tile_output\\' 
	directories = ['h06v02','h06v03','h06v04','h07v02','h07v03','h07v04','h08v02'] #file structure for the NetCDF4 files
	variables = ['tmin']#, 'tmax']
	mosaic_tair = [] # create an empty mosaic to store the outputs and string arrays together
	for i in range(1):
		for v in variables:
			filename = '%s%s\\%s_%s.nc' %(workspace, directories[i],directories[i], v)
			nc_ds = Dataset(filename)
			days = utld.get_days_metadata_dates(num2date(nc_ds.variables['time'][:], units=nc_ds.variables['time'].units))
			tair_trend = twxs.TairTrend(days,1948,2012)
				#tair_agg = twxs.TairAggregate(days)
				#Doing this in one shot will take ~10GB of memory
				#For less memory usage, process in chunks
			tair = nc_ds.variables[v][:]
			tair_ann = tair_trend.get_ann_trend(tair)
			mosaic_tair.append(tair_ann)
			#tair_mon_agg = tair_agg.daily_to_mthly(tair)
			#tair_ann_agg = tair_agg.daily_to_ann(tair)
			#figure out the non-spatial average
			#Output results
	
    #twx_tile_to_gtiff(nc_ds, tair_ann, 'E:\\TOPOWX\\topowx_tile_output\\h06v02\\h06v02_tmin_trend.tiff')