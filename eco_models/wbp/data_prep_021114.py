# -*- coding: utf-8 -*-
"""
Created on Mon Mar 04 18:14:32 2013

@author: tony.chang
#Latest run 01.08.14 for GYE to match GCM extent
#NOTE: must create writename directory in entirety before running, otherwise PRISMtiffwrite will not write file
"""

import numpy as np
#from osgeo import osr
#from osgeo import gdal, gdal_array
from osgeo.gdalconst import GDT_Float32

import gdal 
from gdalconst import *
import osr 

class GridData(object): #initialize function to construct PRISMdata class	
	def __init__(self, year=None, month=None, var = None, model = None, rcp = None, data=None):
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
		self.var = var
		self.model = model
		self.rcp = rcp
		self.data = data
	def mean(self): #method to get the mean of the domain
		return(np.mean(self.data))

def GridMeansExtract(endyear,var,mod,rcp = np.nan): #30 year means extraction
	gdata = [] #List to store all PRISMData object
	if mod =='PRISM':
		workspace = "E:\\PRISM\\30yearnormals\\" + var +"\\"    
	else:
		workspace = "E:\\CMIP5\\GCM\\" + mod + "\\rcp" + str(rcp) + "\\30yearnormals\\" +var +"\\"
	for month in range(1,13):
		filename = workspace + var + "_" + str(endyear-30) + "_" + str(endyear) + "_" + str(month)+ ".tif"
		readfile =  gdal.Open(filename)
		data = np.array(readfile.GetRasterBand(1).ReadAsArray())
		year = str(endyear-30) + '-' + str(endyear)
		x = GridData(year, month, var, mod, rcp,data) #Create instance of PRISMData object
		gdata.append(x) 
		readfile = None #close file
	return(gdata)
	
def HeaderExtract():
	#takes arbitrary PRISM dataset and extracts the header parameters
	filename = "E:\\PRISM\\tmin\\PRISM800m_tmin1895_1.tif" 
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

def Gridcreate():
    header = HeaderExtract()
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
	
def GetFieldData(filename):
	d = np.genfromtxt(filename, delimiter =',', names = True)
	p = np.array([d['y'],d['x'], d['response']])
	return(p.T)

def LookUp(predictor, point): #input of point value as array of [x,y]
	pointx = point[0]
	pointy = point[1]
	h = HeaderExtract()
	csize = h['csize']
	mergedata =[]
	#first check if the point is within the extent of the raster 
	uppery = h['yul']
	lowery = h['yul']-(h['csize']*(h['nrows']+1))
	upperx = h['xul']+(h['csize']*(h['ncols']+1))
	lowerx = h['xul']
	if ((pointy >= uppery) or (pointy <= lowery) or (pointx >= upperx) or (pointx <= lowerx)): 
		return(np.nan)
	#if within the extent, check each if the point is within the bounds of each raster pixel
	for i in range(len(predictor)):
		uppery = predictor[i,0]
		lowery = predictor[i,0]-h['csize']
		upperx = predictor[i,1]+h['csize']
		lowerx = predictor[i,1]
		if ((pointy <= uppery) and (pointy >= lowery) and (pointx <= upperx) and (pointx >= lowerx)): 
			return(i)

def DataIndex(predictor, response):
	indexarray=[]
	for i in range(len(response)):
		p = np.array([response[i,1],response[i,0]])
		val = LookUp(predictor,p)
		if val == None:
			val = np.nan
		indexarray.append(val)
	return(np.array(indexarray))

def DataMerge(predictor, response, index):
	mergeddata = []
	for i in range(len(index)):
		if (index[i] != None):
			if not(np.isnan(index[i])):
				mergeddata.append(np.hstack((response[i], predictor[index[i],2:])))
	return(np.array(mergeddata))

#================main=================#
#Historic climate data prep#
#Field data preperation# 
#can be made into function
#requires a csv file of lat/lon and response
filename = 'E:\\WBP_model\\field_points.csv'
fd = GetFieldData(filename)
varlist = ['tmin', 'tmax', 'ppt', 'aet', 'pet', 'pack', 'soilm', 'vpd']
modellist = ['PRISM']
yearlist = [1980]
head = HeaderExtract()
lat, lon = Gridcreate()
trows = head['nrows']*head['ncols']
for year in yearlist:
	for m in modellist:
		data = np.vstack((np.reshape(lat,trows), np.reshape(lon,trows)))
		labels = ['lat','lon']
		for v in varlist:
			dgrid = GridMeansExtract(year, v, m)
			for i in range(len(dgrid)):
				tdata = np.reshape(dgrid[i].data,trows)
				data = np.vstack((data,tdata))
				l = v + str(i+1)					
				labels.append(l)
		data = data.T
		filename = 'E:\\wbp_model\\fielddata\\' + m +  '_' + str(year-30) + '_' + str(year) + '_data.csv'
		np.savetxt(filename, data, fmt='%.6e', delimiter=',', header = ','.join(labels))
index = DataIndex(data, fd)
mdata = DataMerge(data,fd,index)
mlabels = ['lat','lon','response']
for i in range(2,len(labels)):
	mlabels.append(labels[i])
mdataname = 'E:\\wbp_model\\' + str(year-30) + '_' + str(year) + '_merged_data.csv' 
np.savetxt(mdataname, mdata, fmt='%.6e', delimiter = ',', header = ','.join(mlabels))

#gcm preperation
gcmvarlist = ['tasmin', 'tasmax', 'pr', 'aet', 'pet', 'pack', 'soilm', 'vpd']
#reminder that the units of tasmax and tasmin are in K, and pr are in kg*m2/sec
modellist = ['HadGEM2-ES', 'HadGEM2-CC', 'HadGEM2-AO', 'CNRM-CM5', 'CMCC-CM', 'CESM1-CAM5', 'CESM1-BGC', 'CCSM4', 'CanESM2']
rcplist = [45,85]
daysinmonth = [31,28,31,30,31,30,31,31,30,31,30,31]
head = HeaderExtract()
lat, lon = Gridcreate()
trows = head['nrows']*head['ncols']
for year in range(2040,2100):
	for m in modellist:
		for r in rcplist:
			data = np.vstack((np.reshape(lat,trows), np.reshape(lon,trows)))
			labels = ['lat','lon']
			for v in gcmvarlist:
				dgrid = GridMeansExtract(year, v, m, r)
				for i in range(len(dgrid)):
					tdata = np.reshape(dgrid[i].data,trows)
					if (v == 'tasmin' or v == 'tasmax'):
						tdata = tdata-272.15 #convert from K to C
						if v == 'tasmin':
							vout ='tmin'
						elif v == 'tasmax':
							vout = 'tmax'
					elif (v == 'pr'):	
						tdata = tdata * 86400 * daysinmonth[i] #convert from kg*m2/sec to mm/month
						vout = 'ppt'
					else: 
						vout = v
					data = np.vstack((data,tdata))
					l = vout + str(i+1)					
					labels.append(l)
			data = data.T
			filename = 'E:\\wbp_model\\projections\\' + m + '\\' + m + '_' + str(r) + '_' + str(year-30) + '_' + str(year) + '_data.csv'
			np.savetxt(filename, data, fmt='%.6e', delimiter=',', header = ','.join(labels))




#=====================================#
