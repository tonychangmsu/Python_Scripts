'''
Title: twx_analysis
Date: Created on October 6, 2014
Utility methods for calculating statistical summaries of TopoWx data
Rebuilt this script for the new TOPOWX NetCDF4 files for CONUS
@author: tony.chang
pip 
'''
import os
#set the working directory to one containing twx_sumry_v2
os.chdir('E:\\TOPOWX')

import numpy as np
from util_dates import YEAR,MONTH
import util_dates as utld
from scipy import stats
import osgeo.gdal as gdal
import osgeo.gdalconst as gdalconst
import osgeo.osr as osr
from netCDF4 import Dataset, num2date
import twx_sumry_v2 as twxs

if __name__ == '__main__': 
	
	startyear = 1948
	endyear = 2012
	workspace = 'E:\\TOPOWX\\annual\\'
	var = ['tmin']#,'tmax']
	i = 0
	dataset = []
	for year in range(startyear, endyear+1):
		filename = '%s%s\\%s_%s.nc' %(workspace, var[i], var[i], str(year))
		nc_ds = Dataset(filename)
		dataset.append(nc_ds)
		
	
		days = utld.get_days_metadata_dates(num2date(nc_ds.variables['time'][:], units=nc_ds.variables['time'].units))
		tair_trend = twxs.TairTrend(days,2012)
				#tair_agg = twxs.TairAggregate(days)
				#Doing this in one shot will take ~10GB of memory
				#For less memory usage, process in chunks
		tair = nc_ds.variables[v][:]
		tair_ann = tair_trend.get_ann_trend(tair)
		ymin = nc_ds.variables['lat'][-1]
		ymax = nc_ds.variables['lat'][0]
		xmin = nc_ds.variables['lon'][0]
		xmax = nc_ds.variables['lon'][-1]
		bbox = [xmin, xmax, ymin, ymax]
		mosaic_tair.append([tair_ann, bbox])
			#tair_mon_agg = tair_agg.daily_to_mthly(tair)
			#tair_ann_agg = tair_agg.daily_to_ann(tair)
			#figure out the non-spatial average
			#Output results
	h1 = np.vstack((np.vstack((mosaic_tair[0][0],mosaic_tair[1][0])), mosaic_tair[2][0]))
	h2 = np.vstack((np.vstack((mosaic_tair[3][0],mosaic_tair[4][0])), mosaic_tair[5][0]))
	v = np.hstack((h1,h2))
    #twx_tile_to_gtiff(nc_ds, tair_ann, 'E:\\TOPOWX\\topowx_tile_output\\h06v02\\h06v02_tmin_trend.tiff')