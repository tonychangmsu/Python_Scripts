#summarize the climate data as plots again for use in the research paper
# author: Tony Chang
# Date: 3.31.2014

import numpy as np
import scipy
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import cm
import gdal 
from gdalconst import *
import osr 

#===================================================================================================
#===================================CLASS DEFINITIONS===============================================
#===================================================================================================
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

class AnnualGridData(object):
	#annual climate data summary of PRISMData object to summarize in years only
	def __init__(self, year=None, data=None):
		self.year = year
		self.data = data
	def mean(self):
		return(np.mean(self.data))
		
def header_extract():
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

def grid_extract(byear, eyear,var,mod,rcp = np.nan): #climate data extract
	gdata = [] #List to store all GridData objects
	monthday = [31,28,31,30,31,30,31,31,30,31,30,31]
	if mod =='PRISM':
		workspace = "E:\\PRISM\\" + var +"\\"    
		model = 'PRISM800m'
	else:
		workspace = "E:\\CMIP5\\GCM\\" + mod + "\\rcp" + str(rcp) + "\\" + var + "\\"
	for year in range(byear, eyear+1):
		for month in range(1,13):
			if mod =='PRISM':
				filename = workspace + model + "_" + var + str(year) + "_" + str(month)+ ".tif"
			else: 
				filename = workspace + mod + "_rcp" + str(rcp) + "_" + var + '_' + str(year) + "_" + str(month)+ ".tif"
			readfile =  gdal.Open(filename)
			data = np.array(readfile.GetRasterBand(1).ReadAsArray())
			if mod !='PRISM': #if this is a GCM model
				if var == 'pr':
					outdata = data*60*60*24*monthday[month-1] #kg m2 s-1 to mm
				else:
					outdata = data-273.15 #K to C
			else:
				outdata = data
			x = GridData(year, month, var, mod, rcp,outdata) #Create instance of PRISMData object
			gdata.append(x) 
			readfile = None #close file
	return(gdata)

def annual_grid(data,p='n'):  #generates the PRISM grids at an annual time step
	#if p is 'y' sum the total monthly values rather than generating the monthly mean
	#there is some conversions required for gcm ppt to convert back to mm
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
			anu_year+=data[i].data
			counter +=1
		elif (currentyear != data[i].year):
			if (p=='n'):
				x =AnnualGridData(data[i-1].year,anu_year/counter)
			else:
				x =AnnualGridData(data[i-1].year,anu_year)
			anu_series.append(x)
			counter = 0
			currentyear = data[i].year
			anu_year = np.zeros(np.shape(data[0].data))
			anu_year +=data[i].data
			counter +=1
		i+=1
	#last iteration
	if (p=='n'):
		x =AnnualGridData(data[i-1].year,anu_year/counter)
	else:
		x =AnnualGridData(data[i-1].year,anu_year)
	anu_series.append(x)
	return(anu_series)  

def annual_series(data):
	t = np.arange(data[0].year, data[-1].year+1) #historic year array
	ts = []
	for i in range(len(data)):
		ts.append(np.mean(data[i].data))
	ts = np.array(ts)
	return(ts)

#==================================================================================================#
#temporary functions to reduce memory usage
#==================================================================================================#
def generate_historic():
	#gathers all the historic data and creates the timeseries 0=tmin, 1=tmax, 2=ppt
	byearH = 1895
	eyearH = 2010
	varlist = ['tmin', 'tmax', 'ppt']
	mod = 'PRISM'
	hist_m = []
	for var in varlist:
		hist_m.append(grid_extract(byearH, eyearH, var, mod))
	#generate annual data
	hist_a = []
	ts = []
	for i in range(len(varlist)):
		if i !=2:
			hist_a.append(annual_grid(hist_m[i]))
		else:
			hist_a.append(annual_grid(hist_m[i], p='y'))
		ts.append(annual_series(hist_a[i]))
	return(np.array(ts))

def generate_projection():
	#gcm data
	byearP = 2010
	eyearP = 2099
	varlist = ['tasmin', 'tasmax', 'pr']
	rcplist =[45,85]
	modlist = ['CanESM2','CCSM4','CESM1-BGC','CESM1-CAM5','CMCC-CM','CNRM-CM5','HadGEM2-AO','HadGEM2-CC','HadGEM2-ES']
	proj_m = []
	for mod in modlist:
		proj_rcp = [[],[]]
		for rcp in range(len(rcplist)):
			for var in varlist:
				 proj_rcp[rcp].append(grid_extract(byearP, eyearP,var, mod,rcplist[rcp]))
		proj_m.append(proj_rcp)
	#generate annual data
	ts = []
	for i in range(len(modlist)):
		proj_a = [[],[]]
		for j in range(len(rcplist)):
			for k in range(len(varlist)):
				if k !=2:
					proj_a[j].append(annual_series(annual_grid(proj_m[i][j][k])))
				else:
					proj_a[j].append(annual_series(annual_grid(proj_m[i][j][k], p ='y')))
		ts.append(proj_a)
	return(np.array(ts))

def timeseries_plots(hdata, pdata, var, filename=None):
	#plot the time series for the requested variable
	# var input as a integer value 0 = tmin, 1 = tmax, 2 = ppt, 3 = tmean
	modlist = ['CanESM2','CCSM4','CESM1-BGC','CESM1-CAM5','CMCC-CM','CNRM-CM5','HadGEM2-AO','HadGEM2-CC','HadGEM2-ES']
	ht = np.arange(1895,2011)
	pt = np.arange(2010,2100)
	ft = np.arange(1895,2100)	
	col = ['orange', 'red']
	rcp = ['RCP 4.5', 'RCP 8.5']
	if var ==3:
		hmean = (hdata[0]+hdata[1])/2
		outh = hmean	
		plt.plot(ft, np.ones(len(ft))*np.mean(outh), color = 'grey', ls = '--', lw = 2.5, alpha=0.8, label='Historic mean')
		plt.plot(ht, outh, lw = 2.5, color ='blue', label = 'Historic record (PRISM)') #plot historic data for variable
		for j in range(len(pdata[0])):
			outavgp = (np.mean(pdata[:,j,0], axis=0) + np.mean(pdata[:,j,1], axis=0))/2
			plt.plot(pt,outavgp, color = col[j], lw=2.5, label = rcp[j] + ' ensemble avg.') #projected mean
			for i in range(len(pdata)):
				outp = (pdata[i][j][0] + pdata[i][j][1])/2
				plt.plot(pt, outp, color = col[j], alpha = 0.3)
	else:
		plt.plot(ft, np.ones(len(ft))*np.mean(hdata[var]), color = 'grey', ls = '--', lw = 2.5, alpha=0.8, label='Historic mean')
		plt.plot(ht, hdata[var], lw = 2.5, color ='blue', label = 'Historic record (PRISM)') #plot historic data for variable
		for j in range(len(pdata[0])):
			plt.plot(pt, np.mean(pdata[:,j,var], axis=0), color = col[j], lw=2.5, label = rcp[j] + ' ensemble avg.') #projected mean
			for i in range(len(pdata)):
				plt.plot(pt, pdata[i][j][var], color = col[j], alpha = 0.3)
	plt.grid()
	plt.legend(loc= 'upper left')
	plt.xlabel('Year')
	plt.xlim(1895,2100)
	if var == 2:
		plt.ylabel('Precipitation (mm)')
	else:
		plt.ylabel('Temperature ($^oC$)')
	if filename != None:
		plt.savefig(filename, bbox_inches ='tight')
	return()
#==================================================================================================#
#==========================================MAIN====================================================#
#==================================================================================================#
if __name__ == '__main__':
	hdata = generate_historic()
	pdata = generate_projection()
	plt.rcParams['figure.figsize'] = 14,10
	timeseries_plots(hdata,pdata,3, 'tmean_fig.png') #tmean
	timeseries_plots(hdata,pdata,2, 'ppt_fig.png') #ppt
	