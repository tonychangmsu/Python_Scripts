#data inspection 
#04.15.2014
#Tony Chang

import numpy as np
from matplotlib import pyplot as plt
import gdal
from gdalconst import *
import osr

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
	GCMdata = []
	for cyear in range(beginyear,endyear+1):
		for month in range(1,13):
			if cyear < 2006: #join the variables for the historic and projection
				if v == 'wb':
					workspace = "E:\\CMIP5\\GCM04112014\\" + model + "\\"
					filename = workspace + 'rcp' + str(rcp) + '\\wb\\' + var + '\\' + var + '_' +str(cyear) + '_' +str(month) +'.tif'
				else:
					workspace = "E:\\CMIP5\\GCM\\" + model + "\\"
					filename = workspace + 'historical' + '\\' + v + '\\' + model + '_historical' + '_' + v + '_' +str(cyear) + '_' +str(month) +'.tif'
			else:
				if v == 'wb':
					workspace = "E:\\CMIP5\\GCM04112014\\" + model + "\\"
					filename = workspace + 'rcp' + str(rcp) + '\\wb\\' + var + '\\' + var + '_' +str(cyear) + '_' +str(month) +'.tif'
				else:
					workspace = "E:\\CMIP5\\GCM\\" + model + "\\"
					filename = workspace + 'rcp' + str(rcp) + '\\' + v + '\\' + model + '_rcp' + str(rcp) + '_' + v + '_' +str(cyear) + '_' +str(month) +'.tif'
			readfile = gdal.Open(filename)
			data = np.array(readfile.GetRasterBand(1).ReadAsArray())
			if (v=='pr'):
				multiplier = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
				data = multiplier[month-1] * data * 86400 # convert from kg m^-2 s^-1 to mm/month (kg/m2 ~= mm)
			elif (v =='tasmin' or v == 'tasmax') :
				data = data - 273.15 #convert from K to C
			x = GridData(cyear,month-1,var, data) ###note here that the month index is ranged 0 through 11
			GCMdata.append(x)
			readfile = None #close file
	return(GCMdata)

def annualgrid(data,p='n'):  #generates the PRISM grids at an annual time step
	#p parameter designates ppt data. Calculates the sum for ppt of the year rather than monthly mean
	numyears = int(len(data)/12)
	anu_year = np.zeros(np.shape(data[0].data))
	anu_series = []
	by = data[0].year
	ey = data[-1].year
	currentyear =by
	i=0
	counter =0
	while (i<len(data)):
		if (currentyear == data[i].year):
			anu_year += data[i].data
			counter += 1
		elif (currentyear != data[i].year):
			if (p=='n'):
				x =AnnualGridData(data[i-1].year,anu_year/counter)
			else:
				x =AnnualGridData(data[i-1].year,anu_year)
			anu_series.append(x)
			counter = 0
			currentyear = data[i].year
			anu_year = np.zeros(np.shape(data[0].data))
			anu_year += data[i].data
			counter += 1
		i+=1
	#last iteration
	if (p=='n'):
		x =AnnualGridData(data[i-1].year,anu_year/counter)
	else:
		x =AnnualGridData(data[i-1].year,anu_year)
	anu_series.append(x)
	return(anu_series)  
#==========================================
#open a test file and look inside
var = 'pack'
month = 4
byear = 1980
eyear = 2070
tocheck1 = np.genfromtxt("E:\\wbp_model\\projections_04122014\\CESM1-BGC\\CESM1-BGC_45_"+str(byear)+"_"+str(eyear)+"_data.csv", delimiter = ',', names=True)
#tocheck2 = np.genfromtxt("E:\\wbp_model\\projections_04122014\\CESM1-BGC\\CESM1-BGC_45_2010_2040_data.csv", delimiter = ',', names=True)
tc = tocheck1[var+str(month)]

qa1 = GCMextract(byear,eyear-1, var, 45, 'CESM1-BGC')
ma = []
for i in range(len(qa1)):
	if qa1[i].month == month-1:
		ma.append(qa1[i].data)

mma = np.mean(np.array(ma), axis=0)
r,c = np.shape(mma)
checker = np.reshape(mma, r*c)
z =tc-checker
#everything checks out! =)

#new checker of water balance
var = 'pack'
month = 4
byear = 1980
eyear = 2099
qa1 = GCMextract(byear,eyear-1, var, 45, 'CESM1-BGC')
qa1 = GCMextract(byear,eyear-1, var, 45, 'CESM1-BGC')
ma = []
for i in range(len(qa1)):
	if qa1[i].month == month-1:
		ma.append(np.mean(qa1[i].data))