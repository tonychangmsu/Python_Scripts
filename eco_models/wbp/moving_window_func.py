import os
import numpy as np
import gdal
from gdalconst import *
import osr
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import math
import shapefile
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
#=====================================================================================
#================================CLASS DEFINITIONS====================================
#=====================================================================================

#define Gridded data class for PRISM or GCMs
class GridData(object): 
	#initialize function to construct PRISMdata class	
	def __init__(self, year=None, month=None, var = None, data=None):
		self.year = year
		self.month = month
		self.var = var
		self.data = data
	def mean(self): #method to get the mean of the domain
		return(np.mean(self.data))

class Annualclimatedata(object):
	#annual climate data summary of PRISMData object to summarize in years only
	def __init__(self, year=None, data=None):
		self.year = year
		self.data = data
	def mean(self): #method to get the mean of the domain
		return(np.mean(self.data))
#=====================================================================================
#======================Gridded data extraction functions==============================
#=====================================================================================

def Headerextract(gcm='n'):
	#takes arbitrary PRISM or GCM dataset and extracts the header parameters
	if gcm =='n':
		filename = "E:\\PRISM\\ppt\\PRISM800m_ppt1895_1.tif"    
	elif gcm =='y':
		filename = "E:\\CMIP5\\GCM\\CanESM2\\rcp45\\pr\\CanESSM2_rcp45_pr_2006_1.tif" 
	dataset = gdal.Open(filename, GA_ReadOnly)
	ncols = dataset.RasterXSize
	nrows = dataset.RasterYSize
	bands = dataset.RasterCount
	driver = dataset.GetDriver().LongName
	geotransform = dataset.GetGeoTransform()
	xul = geotransform[0]
	yul = geotransform[3]
	csize = geotransform[1]
	header = {'ncols':ncols, 'nrows':nrows,'bands':bands,'driver':driver, 'xul':xul, 'yul':yul, 'csize':csize}
	return(header) #returns header as directory for readability

def GCMextract(beginyear,endyear,var,rcp,model):
#use the followling list to call models
#gcmlist = ['CanESM2', 'CCSM4', 'CESM1-BGC','CESM1-CAM5', 'CMCC-CM', 'CNRM-CM5', 'HadGEM2-AO', 'HadGEM2-CC', 'HadGEM2-ES']
#model = gcmlist[i]
	if (var == 'ppt'):
		v = 'pr'
	elif (var =='tmin'):
		v = 'tasmin'
	elif (var == 'tmax'):
		v = 'tasmax'
	else:
		v = 'wb'
	workspace = "E:\\CMIP5\\GCM\\" + model + "\\"
	GCMdata = []
	for cyear in range(beginyear,endyear+1):
		for month in range(1,13):
			if cyear < 2006: #join the variables for the historic and projection
				if v == 'wb':
					filename = workspace + 'historical' + '\\wb\\' + var + '\\' + var + '_' +str(cyear) + '_' +str(month) +'.tif'
				else:
					filename = workspace + 'historical' + '\\' + v + '\\' + model + '_historical' + '_' + v + '_' +str(cyear) + '_' +str(month) +'.tif'
			else:
				if v == 'wb':
					filename = workspace + 'rcp' + str(rcp) + '\\wb\\' + var + '\\' + var + '_' +str(cyear) + '_' +str(month) +'.tif'
				else:
					filename = workspace + 'rcp' + str(rcp) + '\\' + v + '\\' + model + '_rcp' + str(rcp) + '_' + v + '_' +str(cyear) + '_' +str(month) +'.tif'
			readfile = gdal.Open(filename)
			data = np.array(readfile.GetRasterBand(1).ReadAsArray())
			if (v=='pr'):
				multiplier = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
				data = multiplier[month-1] * data * 86400 # convert from kg m^-2 s^-1 to mm/month (kg/m2 ~= mm)
			elif (v =='tasmin' or v == 'tasmax') :
				data = data - 273.15 #convert from K to C
			x = GridData(cyear,month-1,var, data)
			GCMdata.append(x)
			readfile = None #close file
	return(GCMdata)

def Gridcreate():
    header = Headerextract()
    lat_list = []
    lon_list = []
    nrows = header['nrows']
    ncols = header['ncols']    
    xmin = header['xul']
    ymin = header['yul']
    csize = header['csize']
    for ystep in range(nrows):
        latstep = ymin-ystep*csize
        lat_list.append(latstep)
    for xstep in range(ncols):
        lonstep = xmin+xstep*csize
        lon_list.append(lonstep)
    lat = np.array([lat_list,]*ncols).transpose()
    lon = np.array([lon_list,]*nrows)
    return(lat,lon)
	
def moving_avg(data, month, n):	
	# captures the moving average for the particular month of interest
	# n years
	d = []
	y = []
	var = data[0].var + str(month)
	for i in range(len(data)):
		if data[i].month == month-1:
			d.append(data[i].data)
			y.append(data[i].year)
	d = np.array(d) #data sorted by specific month of interest
	ma = []
	for j in range(len(d)-n):
		nd = np.mean(d[j:j+n],axis=0)
		yearl= str(y[j]) + '_' + str(y[j]+n)
		ma.append(GridData(yearl, month, var,  nd))
	return(ma)

def write_data(filepath, data):
	lat, lon = Gridcreate()
	nrows, ncols = np.shape(lat)
	size = nrows*ncols
	writedata = []
	writedata.append(np.reshape(lat, size))
	writedata.append(np.reshape(lon,size))
	label = ['lat', 'lon']
	r,c = np.shape(data)
	for i in range(r):
		for j in range(c):
			writedata.append(np.reshape(data[i][j].data, size))
			label.append(data[i][j].var)
	writelabel = ','.join(label)
	np.savetxt(filepath, np.array(writedata).T,fmt='%.6e', delimiter =',', header=writelabel)
	return(print(filepath + ' file written!'))
			
#===============================================================================================================
#===============================================================================================================
#===============================================================================================================

#develop a routine to get the 30 year means for all the GCMs between 2010-2040 
#problem requires a combination between the historical data and projection data

def save_mw_data():
	avg = 30 
	varnames =  ['tmin','tmax','ppt','aet', 'pet', 'pack', 'soilm', 'vpd']
	gcmlist = ['CanESM2', 'CCSM4', 'CESM1-BGC','CESM1-CAM5', 'CMCC-CM', 'CNRM-CM5', 'HadGEM2-AO', 'HadGEM2-CC', 'HadGEM2-ES']
	rcplist = [45,85]
	beginyear = 1980
	endyear = 2010
	#write the data
	#in format, lat, lon, var[0]1.....var[0]12, ....var[n]12 
	outdata = Gridcreate()
	for year in range(beginyear,endyear):
		for mod in gcmlist:
			for rcp in range(len(rcplist)):
				outset=[]
				for var in varnames:
					outvar=[]
					vardata = GCMextract(year, year+avg, var, rcplist[rcp], mod)
					for month in range(1,13):
						outvar.append(moving_avg(vardata, month, avg)[0])
					outset.append(outvar)
				filename = 'E:\\WBP_model\\projections\\' + mod + '\\' + mod + '_' + str(rcplist[rcp]) + '_' + str(year) + '_' + str(year+avg) + '_data.csv'
				write_data(filename, outset)
	return()
	
#write data
save_mw_data()

