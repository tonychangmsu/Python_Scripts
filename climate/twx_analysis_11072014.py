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

class GridData(object): #initialize function to construct PRISMdata class	
	def __init__(self, year=None, month=None, days = None, var = None, model = None, rcp = None, data=None):
		self.year = year
		self.month = month
		if month == 14: #month 14 in PRISM data 
			self.season = "ALL"
		elif (month < 3 or month == 12):
			self.season = "Win"
		elif month < 6:
			self.season = "Spr"
		elif month < 9:
			self.season = "Sum"
		else:
			self.season = "Fal"
		self.days = days
		self.var = var
		self.model = model
		self.rcp = rcp
		self.data = data
	def mean(self): #method to get the mean of the domain
		return(np.mean(self.data))

class AnnualGridData(object):
	#annual climate data summary of PRISMData object to summarize in years only
	def __init__(self, year=None, data=None):
		self.year = year
		self.data = data
	def mean(self):
		return(np.mean(self.data))

def tiffWrite(data,Nx,Ny,cellsize,yul,xll, path, name):   
	fileformat = "GTiff"
	nbands = 1
	driver = gdal.GetDriverByName(fileformat)
	geotransform = [xll, cellsize,0.0,yul, 0.0, -cellsize]
	srs = osr.SpatialReference()
	writename = path + name
	outDs = driver.Create(writename, Nx, Ny, nbands, gdal.GDT_Float32)
	outDs.SetGeoTransform(geotransform)
	srs.SetWellKnownGeogCS("WGS72")
	outDs.SetProjection(srs.ExportToWkt())
	for band in range(nbands):
		outBand = outDs.GetRasterBand(band+1)
		outBand.SetNoDataValue(-9999)
		outBand.WriteArray(data,0,0)
	outDs = None
	return(print(writename + " filebuilt!\n"))

def writeData(dataset):
	for i in range(len(dataset)):
		name = "TOPOWX_GYE_%s_%s_%s.tif"%(dataset[i].var, dataset[i].month, dataset[i].year) 
		path = "E:\\TOPOWX\\GYE\\%s\\"%(dataset[i].var)
		Ny,Nx = np.shape(dataset[0].data)
		csize = 0.00833333333
		xll = -112.39583333838
		yul = 46.19583332448
		tiffWrite(dataset[i].data, Nx,Ny,csize,yul,xll,path,name)
	return()
		
if __name__ == '__main__': 
	csize = 0.00833333333
	xmax = -108.19583334006; xmin = -112.39583333838; ymin = 42.270833326049996; ymax = 46.19583332448 # GYE bounds
	AOA = [xmin, ymin, xmax, ymax] #specify the bounds for the FIA data
	filename = '%s%s\\%s_%s.nc' %(workspace, 'tmin', 'tmin', 1948)
	i = 0
	nc_ds = Dataset(filename)
	max_x_i = np.where(nc_ds.variables['lon'][:]>=AOA[2])[0][0]
	min_x_i = np.where(nc_ds.variables['lon'][:]>=AOA[0])[0][0]
	min_y_i = np.where(nc_ds.variables['lat'][:]<=AOA[3])[0][0]
	max_y_i = np.where(nc_ds.variables['lat'][:]<=AOA[1])[0][0]
	
	startyear = 1948
	endyear = 2012
	workspace = 'E:\\TOPOWX\\annual\\'
	var = ['tmin','tmax']
	
	for v in var:
		dataset = []
		for year in range(startyear, endyear+1):
			filename = '%s%s\\%s_%s.nc' %(workspace, v, v, str(year))
			nc_ds = Dataset(filename)
			days = utld.get_days_metadata_dates(num2date(nc_ds.variables['time'][:], units=nc_ds.variables['time'].units))
			month = 1
			monthdata = np.zeros((max_y_i-min_y_i, max_x_i-min_x_i))
			for i in range(len(nc_ds.variables[v])): #solve for the monthly average
				if days[i][0].month == month:
					monthdata += nc_ds.variables[v][i][min_y_i:max_y_i,min_x_i:max_x_i].data
				else: #if going to the next month
					dataset.append(GridData(days[i][0].year, days[i-1][0].month, days[i-1][0].day, v, 'TOPOWX', 'None', monthdata/days[i-1][0].day))
					month = days[i][0].month #shift the month counter up to the new month
					monthdata = nc_ds.variables[v][i][min_y_i:max_y_i,min_x_i:max_x_i].data #reassign monthdata to store the first new day
			dataset.append(GridData(days[i][0].year, days[i][0].month, days[i][0].day, v, 'TOPOWX', 'None', monthdata/days[i][0].day)) #last month save
		writeData(dataset)	#write out the dataset and save it
		

	
		
	
		days = utld.get_days_metadata_dates(num2date(nc_ds.variables['time'][:], units=nc_ds.variables['time'].units))
		tair_trend = twxs.TairTrend(days,2012)
				#tair_agg = twxs.TairAggregate(days)
				#Doing this in one shot will take ~10GB of memory
				#For less memory usage, process in chunks
		tair = nc_ds.variables[var[0]][:]
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