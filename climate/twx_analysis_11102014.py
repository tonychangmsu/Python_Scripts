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
from datetime import date
import shapefile
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import statsmodels.api as sm
import ogr
import geotool as gt

class GridData(object): #initialize function to construct GRID data class	
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

def annualGrid(data,p='n'):  #generates the grids at an annual time step
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

def annualSeries(data): #input annual grid data
	t = np.arange(data[0].year, data[-1].year+1) #historic year array
	ts = []
	for i in range(len(data)):
		ts.append(np.mean(data[i].data))
	ts = np.array(ts)
	return(ts)
	
def topoExtract():
	#since shape of the GCM is different from the PRISM extent, a modified grid is required if gcm parameter is set to 'y'
	aspectPath = "E:\\gye_topo\\aspect_800m.tif"   
	slopePath = "E:\\gye_topo\\slope_800m.tif"   
	elevPath = "E:\\gye_topo\\dem_800m.tif"   
	ds = gdal.Open(aspectPath)
	aspect = np.array(ds.GetRasterBand(1).ReadAsArray())
	ds = gdal.Open(slopePath)
	slope = np.array(ds.GetRasterBand(1).ReadAsArray())
	ds = gdal.Open(elevPath)
	elev = np.array(ds.GetRasterBand(1).ReadAsArray())
	ds = None #close files
	return(aspect[:,:], slope[:,:], elev[:,:]) 

def lstFit(data):
	#least squares fit of the data at the individual cell level
	nrows, ncols = (np.shape(data[0].data))
	n = len(data)
	mu = np.zeros((nrows,ncols))
	for i in range(n):
		mu += data[i].data
	mu = mu/n
	tmu = np.sum(np.arange(n)+1)/n
	tmu = np.ones((nrows,ncols))*tmu
	Sxx = np.zeros((nrows,ncols))
	Sxy = np.zeros((nrows,ncols))
	SST = np.zeros((nrows,ncols))
	SSW = np.zeros((nrows,ncols))
	for i in range(n):
		dt = data[i].data
		Sxx += (((np.ones((nrows,ncols))*(i+1))-tmu)**2)
		Sxy += (dt-mu)*((np.ones((nrows,ncols))*(i+1))-tmu)
	beta1 = Sxy/Sxx
	beta0 = mu - (beta1*tmu)
	return(beta1,beta0)	
	
def climateMean(data, p='n'):
	#solves the period mean for plotting purposes
	adata = annualGrid(data, p)
	nyears = len(adata)
	periodmean = np.zeros(np.shape(data[0].data))
	for i in range(nyears):
		periodmean += adata[i].data
	periodmean = periodmean/nyears
	return(periodmean)
	
def hillShade(data,scale=10.0,azdeg=165.0,altdeg=45.0):
	# takes in the elevation grid (data) and generates a hillshade matrix for plotting
	# convert alt, az to radians
	az = azdeg*np.pi/180.0
	alt = altdeg*np.pi/180.0
	# gradient in x and y directions
	dx, dy = np.gradient(data/float(scale))
	slope = 0.5*np.pi - np.arctan(np.hypot(dx, dy))
	aspect = np.arctan2(dx, dy)
	intensity = np.sin(alt)*np.sin(slope) + np.cos(alt)*np.cos(slope)*np.cos(-az - aspect - 0.5*np.pi)
	intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
	return(intensity)	

def drawArea(AOA):
	#plots the boundary for natural resource of interest
	minx = AOA[0] 
	maxx = AOA[1]
	miny = AOA[2]
	maxy = AOA[3]
	sf = shapefile.Reader("d:\\chang\\gis_data\\gye_shapes\\gye.shp") #change the shapefile location here!
	recs    = sf.records()
	shapes  = sf.shapes()
	Nshp    = len(shapes)
	cns     = []
	for nshp in range(Nshp):
		cns.append(recs[nshp][1])
	cns = np.array(cns)
	cma    = cm.get_cmap('Dark2')
	cccol = cma(1.*np.arange(Nshp)/Nshp)
	ax = plt.subplot(111)
	for nshp in range(Nshp):
		ptchs   = []
		pts     = np.array(shapes[nshp].points)
		prt     = shapes[nshp].parts
		par     = list(prt) + [pts.shape[0]]
		for pij in range(len(prt)):
			ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))
			ax.add_collection(PatchCollection(ptchs,facecolor ='None',edgecolor='r', linewidths=2))#facecolor=cccol[nshp,:]
	ax.set_xlim(minx,maxx)
	ax.set_ylim(miny,maxy)
	return()

def plotElevation(AOA):
	#plots the hillshade given the topography and area of interest as a background for plots
	a,s,ele = Topoextract()
	hill = hillshade(ele)
	im = plt.imshow(hill, cmap = cm.Greys_r, extent =AOA)
	#im2 = plt.imshow(ele, cmap = cm.Spectral, alpha= 0.7, extent =ae)
	plt.xlabel('Longitude (DD)')
	plt.ylabel('Latitude (DD)')
	#cb = plt.colorbar(im2)
	#cb.set_label('Elevation (m)')
	plt.grid(alpha =0.4)	
	return()
	
def plotGrad(pdata,AOA, p='n'):
	#plots the PRISM gradients with hillshade and resource boundary
	plotelevation(AOA)
	dzdt = temporalgradient(pdata,p)
	if p=='y':
		dz= dzdt*10
		label='mm/decade' #change the label here depending on the variable type
	else:
		dz=dzdt*10*9/5
		label=r'$^oF/decade$' #change the label here depending on the variable type
	ax = plt.imshow(dz, extent = AOA, alpha =0.6)
	cb = plt.colorbar(ax)
	cb.set_label(label) #change the label here depending on the variable type
	drawArea(AOA)
	return()
	
def shapeMask(sname,rname):
	#takes a shapefile and generates a boolean mask from it (assumes single layer)
	raster_ref = gdal.Open(rname)
	sh_ds = shapefile.Reader(sname)
	shape = sh_ds.shapes()
	pnts = np.array(shape[0].points).T
	fextent = gt.getFeatureExtent(pnts)
	mask = gt.rasterizer(pnts, raster_ref)
	mask_bool = np.where(mask==1, True, False)
	return(mask_bool, pnts, fextent)
	#============================================================================================

if __name__ == '__main__': 
	workspace = 'E:\\TOPOWX\\annual\\'
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
	var = ['tmin','tmax']
''' SUBSETTING PROCEDURES!	
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

	#subset the standard error for the 1980-2010 normals
	workspace = 'E:\\TOPOWX\\'
	var = ['tmin','tmax']
	for v in var:
		dataset = []
		filename = '%snormals_%s.nc'%(workspace, v)
		nc_ds = Dataset(filename)
		days = utld.get_days_metadata_dates(num2date(nc_ds.variables['time'][:], units=nc_ds.variables['time'].units))
		vname = '%s%s'%(v,'_se')
		for i in range(len(nc_ds.variables[vname])): #solve for the monthly average
			monthdata = nc_ds.variables[vname][i][min_y_i:max_y_i,min_x_i:max_x_i].data
			dataset.append(GridData('1980_2010', i+1, None, vname, 'TOPOWX', 'None', monthdata))
		writeData(dataset)	#write out the dataset and save it
'''		
	#compare to PRISM?
	topowx=[[],[]]
	prism=[[],[]]
	atopowx=[] #annual data storage
	aprism=[] #annual data storage
	atopogrid = []
	aprismgrid = []
	t = np.arange(startyear, endyear-1)
	
	for v in range(0,2):
		days = [31,28,31,30,31,30, 31, 31, 30, 31, 30,31]
		for year in range(startyear, endyear-1): #PRISM only goes to 2010
			for month in range(1,13):
				fname_prism = 'E:\\PRISM\\%s\\PRISM800m_%s%s_%s.tif' %(var[v], var[v], year, month)
				ds = gdal.Open(fname_prism).ReadAsArray()
				prism[v].append(GridData(year, month, days[month-1], var[v], 'PRISM', 'None', ds))			
				fname_topowx = 'E:\\TOPOWX\\GYE\\%s\\TOPOWX_GYE_%s_%s_%s.tif' %(var[v], var[v], month, year)
				ds = gdal.Open(fname_topowx).ReadAsArray()
				topowx[v].append(GridData(year, month, days[month-1], var[v], 'PRISM', 'None', ds))			
		atopogrid.append(annualGrid(topowx[v]))
		aprismgrid.append(annualGrid(prism[v]))
		atopowx.append(annualSeries(annualGrid(topowx[v])))
		aprism.append(annualSeries(annualGrid(prism[v])))

	atopowx = np.array(atopowx)
	aprism = np.array(aprism)
''' #completed this part of the analysis 11/11/2014 @t.chang
	#long term PRISM data?
	prism_long = [[],[]]
	aprism_long_topo = []
	aprism_long = []
	days = [31,28,31,30,31,30, 31, 31, 30, 31, 30,31]
	startyear_long = 1895
	for v in range(0,2):		
		for year in range(startyear_long, endyear-1): #PRISM only goes to 2010
			for month in range(1,13):
				fname_prism = 'E:\\PRISM\\%s\\PRISM800m_%s%s_%s.tif' %(var[v], var[v], year, month)
				ds = gdal.Open(fname_prism).ReadAsArray()
				prism_long[v].append(GridData(year, month, days[month-1], var[v], 'PRISM', 'None', ds))			
		aprism_long_topo.append(annualGrid(prism_long[v]))
		aprism_long.append(annualSeries(annualGrid(prism_long[v])))
	aprism_long = np.array(aprism_long)
	t_long = np.arange(startyear_long, endyear-1)
	
	#plot the results
	Fatopowx = atopowx*(9/5)+32
	Faprism_long = aprism_long*(9/5)+32
	my_dpi = 96
	plt.rcParams['figure.figsize'] = 12,8
	plt.figure(figsize==(800/my_dpi, 800/my_dpi), dpi=my_dpi)
	plt.plot(t, np.mean(Fatopowx, axis=0),label='TopoWx ($\mu = %0.1f^oF$)'%(np.mean(Fatopowx)), color = 'red', lw = 2,alpha = 0.8)
	plt.plot(t_long,np.mean(Faprism_long, axis=0), label='PRISM ($\mu = %0.1f^oF$)'%(np.mean(Faprism_long)), color = 'blue', lw= 2, alpha = 0.8)
	plt.legend(loc = 'upper left')
	plt.grid()
	plt.xlabel('Year')
	plt.ylabel('Mean temperature ($^oF$)')
	plt.plot(t, np.ones(len(t))*np.mean(Fatopowx), color = 'red', ls = '--')
	plt.plot(t_long, np.ones(len(t_long))*np.mean(Faprism_long), color = 'blue', ls='--')
	plt.savefig('D:\\CHANG\\PhD_Material\\Manuscripts\\YS_paper\\figs\\current_plot_%s.png'%(date.today().strftime('%d%m%y')), bbox_inches='tight', dpi = my_dpi)
	
	#repeat for ppt for PRISM
	prism_ppt = []
	days = [31,28,31,30,31,30, 31, 31, 30, 31, 30,31]
	startyear_long = 1895
	for year in range(startyear_long, endyear-1): #PRISM only goes to 2010
		for month in range(1,13):
			fname_prism = 'E:\\PRISM\\%s\\PRISM800m_%s%s_%s.tif' %('ppt', 'ppt', year, month)
			ds = gdal.Open(fname_prism).ReadAsArray()
			prism_ppt.append(GridData(year, month, days[month-1], 'ppt', 'PRISM', 'None', ds))			
	aprism_ppt_topo = annualGrid(prism_ppt,p='y')
	aprism_ppt = annualSeries(annualGrid(prism_ppt,p='y'))
	t_long = np.arange(startyear_long, endyear-1)
	Iaprism_ppt = aprism_ppt*0.0393701
	plt.figure(figsize==(800/my_dpi, 800/my_dpi), dpi=my_dpi)
	plt.plot(t_long,Iaprism_ppt, label='PRISM ($\mu = %0.1f in$)'%(np.mean(Iaprism_ppt)), color = 'green', lw= 2, alpha = 0.8)
	plt.legend(loc = 'upper left')
	plt.grid()
	plt.xlabel('Year')
	plt.ylabel('Mean annual precipitation ($inches$)')
	plt.plot(t_long, np.ones(len(t_long))*np.mean(Iaprism_ppt), color = 'green', ls='--')
	plt.savefig('D:\\CHANG\\PhD_Material\\Manuscripts\\YS_paper\\figs\\current_plot_ppt%s.png'%(date.today().strftime('%d%m%y')), bbox_inches='tight', dpi = my_dpi)
'''
	####################################################################################################################################
	####################################################################################################################################
	####################################################################################################################################
	
''' Plots including the lowess curve	
	lowess = sm.nonparametric.lowess
	topowx_lowess = lowess(atopowx.mean(axis=0),t) #lowess results
	prism_lowess = lowess(aprism.mean(axis=0),t)
	topowx_statout  = stats.linregress(t, atopowx.mean(axis=0)) #OLS results
	prism_statout  = stats.linregress(t, aprism.mean(axis=0))
	
	plt.rcParams['figure.figsize'] = 10,8
	plt.plot(t, atopowx.mean(axis=0),label='TOPOWX ($%0.2f ^oC/decade$)'%(topowx_statout[0]*10), color='red')
	plt.plot(t,aprism.mean(axis=0), label='PRISM ($%0.2f ^oC/decade$)' %(prism_statout[0]*10), color='blue')

	plt.plot(t, t*topowx_statout[0]+topowx_statout[1], ls ='--', color = 'red')
	plt.plot(t,t*prism_statout[0]+prism_statout[1], ls = '--', color = 'blue')

	plt.plot(t,topowx_lowess[:,1], ls=':', color='red')
	plt.plot(t,prism_lowess[:,1], ls=':', color='blue')

	plt.legend(loc='upper left')
'''	
	#check the cell by cell trend analysis for topowx
	topowx=[[],[]]
	atopowx=[] #annual data storage
	atopogrid = []
	startyear = 1948
	endyear = 2013
	t = np.arange(startyear, endyear)
	var = ['tmin', 'tmax']
	for v in range(0,2):
		days = [31,28,31,30,31,30, 31, 31, 30, 31, 30,31]
		for year in range(startyear, endyear): #PRISM only goes to 2010
			for month in range(1,13):
				fname_topowx = 'E:\\TOPOWX\\GYE\\%s\\TOPOWX_GYE_%s_%s_%s.tif' %(var[v], var[v], month, year)
				ds = gdal.Open(fname_topowx).ReadAsArray()
				topowx[v].append(GridData(year, month, days[month-1], var[v], 'PRISM', 'None', ds))			
		atopogrid.append(annualGrid(topowx[v]))
		atopowx.append(annualSeries(annualGrid(topowx[v])))
	atopowx = np.array(atopowx)
	
	topowx_trend = []
	for v in range(2):
		topowx_trend.append(lstFit(atopogrid[v])[0])
		prism_trend.append(lstFit(aprismgrid[v])[0])
		diff_mask.append(lstFit(aprismgrid[v])[0]-lstFit(atopogrid[v])[0])
	topowx_trend = np.array(topowx_trend)
	prism_trend = np.array(prism_trend)
	diff_mask = np.array(diff_mask)
	topowx_trend_mu = np.mean(topowx_trend, axis=0)
	prism_trend_mu = np.mean(prism_trend, axis=0)
	diff_mask_mu = np.mean(diff_mask, axis=0)
	
	#first thing is to plot the dem, topowx, prism, and difference in trends
	#first tmin...
	#============================================================================================
	a,s,elevation = topoExtract()
	hill = hillShade(elevation)
	Felevation = elevation * 3.28084
	xmin = -112.39583333837999 #112 23 45
	xmax = -108.19583334006 #108 11 45
	ymin = 42.279166659379996 #42 16 45
	ymax = 46.195833324479999 #46 11 45
	ae = [xmin, xmax,ymin, ymax]

	i = 0 #plot tmin first
	plt.subplot2grid((2,4),(i,0))
	plt.imshow(hill, cmap = cm.gray, extent = ae)
	plt.imshow(Felevation, cmap=cm.gray, alpha=0.7, extent = ae)
	plt.colorbar(shrink = 0.5)
	
	Fprism_trend = prism_trend*(9/5)*10
	Ftopowx_trend = topowx_trend*(9/5)*10
	
	Fdiff_mask = diff_mask*(9/5)*10
	thr = np.array([np.std(Fdiff_mask[0])*2,np.std(Fdiff_mask[1])*2])
	binary_diff = np.array([np.where(np.abs(Fdiff_mask[0])>thr[0],0,np.nan),np.where(np.abs(Fdiff_mask[1])>thr[1],0,np.nan)])
	var = ['T_{min}', 'T_{max}', 'T_{mean}']
	plt.figure(figsize==(800/my_dpi, 800/my_dpi), dpi=my_dpi)
	for i in range(2):
		plt.subplot2grid((2,3),(i,0))
		plt.imshow(Fprism_trend[i], extent = ae, vmin =-0.7, vmax=0.8)
		cbar = plt.colorbar(shrink = 0.5)
		cbar.set_label('$^oF/decade$')
		plt.title('PRISM $d%s/dt$'%var[i])
		plt.subplot2grid((2,3),(i,1))
		plt.imshow(Ftopowx_trend[i], extent = ae,vmin =-0.7, vmax=0.8)
		cbar = plt.colorbar(shrink = 0.5)
		cbar.set_label('$^oF/decade$')
		plt.title('TopoWx $d%s/dt$'%var[i])
		plt.subplot2grid((2,3),(i,2))
		plt.imshow(np.abs(binary_diff[i]), extent = ae, cmap = cm.gray)
		cbar = plt.colorbar(shrink = 0.5)
		cbar.set_label('|PRISM minus TopoWx| $> 2 \sigma$ $(\sigma = %0.1f)$'%thr[i])
		plt.title('PRISM minus TopoWx $d%s/dt$'%var[i])
	plt.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.05)	
	plt.savefig('D:\\CHANG\\PhD_Material\\Manuscripts\\YS_paper\\figs\\historic_trend%s.png'%(date.today().strftime('%d%m%y')), bbox_inches='tight', dpi = my_dpi)
	
	#generate the means
	i = 2
	Fprism_trend_mu = prism_trend_mu*(9/5)*10
	Ftopowx_trend_mu = topowx_trend_mu*(9/5)*10
	Fdiff_mask_mu = diff_mask_mu*(9/5)*10
	thr_mu = np.std(Fdiff_mask_mu)*2
	binary_diff_mu = np.where(np.abs(Fdiff_mask_mu)>thr_mu,0,np.nan)
	plt.subplot2grid((1,3),(0,0))
	plt.imshow(Fprism_trend_mu, extent = ae, vmin = -0.3, vmax =0.6)
	cbar = plt.colorbar(shrink = 0.3)
	cbar.set_label('$^oF/decade$')
	plt.title('PRISM $d%s/dt$'%var[i])
	plt.subplot2grid((1,3),(0,1))
	plt.imshow(Ftopowx_trend_mu, extent = ae, vmin = -0.3, vmax =0.6)
	cbar = plt.colorbar(shrink = 0.3)
	cbar.set_label('$^oF/decade$')
	plt.title('TopoWx $d%s/dt$'%var[i])
	plt.subplot2grid((1,3),(0,2))
	plt.imshow(np.abs(binary_diff_mu), extent = ae, cmap = cm.gray)
	cbar = plt.colorbar(shrink = 0.2)
	cbar.set_label('$> 2 \sigma$')
	plt.title('PRISM minus TopoWx $d%s/dt$'%var[i])
	plt.tight_layout(pad=0.1, w_pad=0.05, h_pad=0.05)	
	
	#generate the means for precipitation? No, spatially explicit precipitation likely wrong
	#consider the standard error for the region annually?
	se_data = [[],[]]
	year = '1980_2010'
	var = ['tmin_se', 'tmax_se']
	for v in range(2):
		for i in range(1,13):
			fname_topose = 'E:\\TOPOWX\\%s\\TOPOWX_GYE_%s_%s_%s.tif' %(var[v], var[v], i, year)
			ds = gdal.Open(fname_prism).ReadAsArray()
			se_data[v].append(GridData(year, i, None, var[v], 'TopoWx', 'None', ds))		
	
	se_t1 = np.mean(np.abs(se_data[0][0].data))
	se_t2 = np.mean(np.abs(se_data[0][0].data))*2

	binary_se1 = np.where(np.abs(se_data[0][0].data)>se_t1, 0,np.nan)
	binary_se2 = np.where(np.abs(se_data[0][0].data)>se_t2, 0,np.nan)
	
	plt.imshow(Ftopowx_trend_mu, cmap =cm.autumn_r, extent = ae); 
	cbar = plt.colorbar(orientation='horizontal')
	cbar.set_label('$^oF/decade$', fontsize=32)
	cbar.ax.tick_params(labelsize=30) 
	#plt.imshow(binary_se1, cmap=cm.gray, alpha=0.3, extent = ae)
	#plt.imshow(binary_se2, cmap=cm.gray, alpha=0.7, extent = ae)
	plt.savefig('D:\\chang\\phd_material\\manuscripts\\ys_paper\\figs\\tmean_se.png', bbox_inches='tight')
	plt.title('TopoWx $d%s/dt$'%('T_{mean}'))
	
''' Writing out trend tiffs
	#write out the tiffs
	Ny,Nx = np.shape(Ftopowx_trend_mu)
	csize = 0.00833333333
	xll = -112.39583333838
	yul = 46.19583332448
	outpath = 'E:\\TOPOWX\\Analysis\\out\\'
	tiffWrite(Ftopowx_trend_mu, Nx, Ny, csize, yul, xll, outpath, "Ftopowx_trend_tmean.tif")
	tiffWrite(Ftopowx_trend[0], Nx, Ny, csize, yul, xll, outpath, "Ftopowx_trend_tmin.tif")
	tiffWrite(Ftopowx_trend[1], Nx, Ny, csize, yul, xll, outpath, "Ftopowx_trend_tmax.tif")
	tiffWrite(Fprism_trend_mu, Nx, Ny, csize, yul, xll, outpath, "Fprism_trend_tmean.tif")
	tiffWrite(Fprism_trend[0], Nx, Ny, csize, yul, xll, outpath, "Fprism_trend_tmin.tif")
	tiffWrite(Fprism_trend[1], Nx, Ny, csize, yul, xll, outpath, "Fprism_trend_tmax.tif")
'''

####Get the GRTE shapefile and subset the trend data
	shps = ['Absaroka_Beartooth', 'Bridger', 'Fitzpatrick', 'Gros_Ventre', 'grte', 'gye', 'Jedediah_Smith', 'North_Absaroka', 'Popo_Agie', 'Teton', 'Washakie', 'yell']
	rname = 'E:\\TOPOWX\\GYE\\tmax\\TOPOWX_GYE_tmax_1_1948.tif' #abitrary reference raster file
	masks = []
	for s in shps:
		if (s == 'grte' or s == 'gye' or s == 'yell'):
			sname = 'E:\\topowx\\Analysis\\shp\\%s\\%s.shp'%(s,s)
		else:
			sname = 'E:\\topowx\\Analysis\\shp\\%s\\%s_p.shp'%(s,s)
		masks.append(shapeMask(sname, rname)[0])
'''	
	sname = 'E:\\topowx\\Analysis\\shp\\yell.shp'
	sh_ds = shapefile.Reader(sname)
	shape = sh_ds.shapes()
	y_pnts = np.array(shape[0].points).T
	y_fextent = gt.getFeatureExtent(y_pnts)
	raster_ref = gdal.Open('E:\\TOPOWX\\GYE\\tmax\\TOPOWX_GYE_tmax_1_1948.tif')
	mask = gt.rasterizer(y_pnts, raster_ref)
	
	yell_mask_bool = np.where(mask==1, True, False)
	yi = np.where(yell_mask_bool == True)
	yell_bounds = np.array([yi[0][0],yi[0][-1],yi[1][0],yi[1][-1]]) #(ymax, ymin, xmin, xmax) #expand area by 1
	#plt.imshow(ma.masked_where(~yell_mask_bool,Ftopowx_trend_mu)[yell_bounds[0]:yell_bounds[1],yell_bounds[2]:yell_bounds[3]], extent = y_ae)
	
	sname = 'E:\\topowx\\Analysis\\shp\\grte2.shp'
	sh_ds = shapefile.Reader(sname)
	shape = sh_ds.shapes()
	g_pnts = np.array(shape[0].points).T
	g_fextent = gt.getFeatureExtent(g_pnts)
	mask = gt.rasterizer(g_pnts, raster_ref)
	
	grte_mask_bool = np.where(mask==1, True, False)
	gi = np.where(grte_mask_bool == True)
	grte_bounds = np.array([gi[0][0],gi[0][-1],gi[1][0],gi[1][-1]]) #(ymax, ymin, xmin, xmax)
	
	#combine the yellowstone and grand_teton shapes
	yell_grte_mask_bool = yell_mask_bool + grte_mask_bool
	
	sname = 'E:\\topowx\\Analysis\\shp\\gye.shp'
	sh_ds = shapefile.Reader(sname)
	shape = sh_ds.shapes()
	gye_pnts = np.array(shape[0].points).T
	gye_fextent = gt.getFeatureExtent(gye_pnts)
	mask = gt.rasterizer(gye_pnts, raster_ref)
	
	gye_mask_bool = np.where(mask==1, True, False)
	gyei = np.where(gye_mask_bool == True)
	gye_bounds = np.array([gyei[0][0],gyei[0][-1],gyei[1][0],gyei[1][-1]]) #(ymax, ymin, xmin, xmax)
	
	#plt.imshow(ma.masked_where(~yell_mask_bool,Ftopowx_trend_mu), extent = ae)
	#plt.imshow(ma.masked_where(~grte_mask_bool,Ftopowx_trend_mu), extent = ae)
	plt.imshow(ma.masked_where(~gye_mask_bool,Ftopowx_trend_mu), extent = ae, cmap = cm.autumn_r)
	cbar = plt.colorbar()
	cbar.set_label('$^oF/decade$', fontsize = 18)
	plt.plot(y_pnts[0], y_pnts[1], lw = 2)
	plt.plot(g_pnts[0], g_pnts[1], lw = 2)
	plt.plot(gye_pnts[0], gye_pnts[1], lw = 2)
	plt.grid()
'''	
#AREA MASKS
	#compare all the areas:
	a_areas = ['GYE', 'YELL/GRTE NP', 'Absaroka/Beartooth/N. Absaroka WA', 'Teton/Washake WA', 'Bridger/Fitzpatrick/Popo Agie WA']
	area_masks = [masks[5], masks[4]+masks[-1], masks[0]+masks[7], masks[-3]+masks[-2], masks[1]+masks[2]+masks[3]]
	topo_summary = []
	data_subset = Ftopowx_trend_mu
	for i in range(len(area_masks)):
		topo_summary.append(data_subset[area_masks[i]]) #can change the dataset here
	topo_summary = np.array(topo_summary)
	
'''	### PLOT BY AREA
	#plot!
	my_dpi = 96
	plt.figure(figsize==(800/my_dpi, 800/my_dpi), dpi=my_dpi)
	ax = plt.subplot(111)
	bp = ax.boxplot(topo_summary, notch=0 ,vert=1, whis=1.5, patch_artist=True)
	boxColors = ['black', 'blue', 'lightblue', 'green', 'yellow']
	for patch, color in zip(bp['boxes'], boxColors):
		patch.set_facecolor(color)
	plt.setp(bp['medians'], color='red', lw =2)
	plt.setp(bp['whiskers'], color='black')
	plt.setp(bp['fliers'], color='red', marker='+')
	plt.xticks(np.arange(1,6),a_areas, fontsize = 11)
	plt.ylabel('$^oF/decade$', fontsize=16)
	ax.yaxis.grid(alpha = 0.8)
	plt.savefig('E:\\topowx\\analysis\\out\\box_tmean.png', bbox_inches='tight', dpi = my_dpi)
	
	#need to get the colorbar for map making
	plt.imshow(elevation*3.28084, cmap = cm.BrBG)
	cbar = plt.colorbar(orientation= 'horizontal')
	cbar.set_label('$Ft$', fontsize = 32)
	cbar.ax.tick_params(labelsize=30) 


	plt.imshow(prism_ppt_trend*0.0393701*10, cmap= cm.Blues)
	cbar = plt.colorbar(orientation= 'horizontal')
	cbar.set_label('$in/decade$', fontsize = 32)
	cbar.ax.tick_params(labelsize=30) 
	
	plt.imshow(Ftopowx_trend_mu, cmap= cm.autumn_r)
	cbar = plt.colorbar(orientation= 'horizontal')
	cbar.set_label('$^oF/decade$', fontsize = 32)
	cbar.ax.tick_params(labelsize=30) 
'''
	#now solve for the means for tmean, tmax, and tmin for all these zones. Note that precipitation is highly variables and difficult to interpolate therefore was not analyzed for the sub-domain level analysis
	#first get topowx_tmean_1948_2012, topowx_tmax_1948_2012,topowx_tmax_1948_2012, topowx_tmean_trend_1948_2012, and prism_ppt_1948_2010, prism_ppt_trend_1948_2010
	#use the masks
	#get the statistical summaries, and generate the latex table
''' GETTING DATA
	#first get all the data
	topowx_all_tmin = climateMean(topowx[0])
	topowx_all_tmax = climateMean(topowx[1])
	prism_all_ppt = climateMean(prism_ppt[636:], p='y') #prism from 1948-2010
	prism_all_ppt_long = climateMean(prism_ppt, p='y')
	
	#generate the topowx mean monthly
	topowx_tmean = []
	for i in range(len(topowx[0])):
		topowx_tmean.append(GridData(topowx[0][i].year, topowx[0][i].month, topowx[0][i].days, 'tmean', 'topowx', 'None', (topowx[0][i].data+topowx[1][i].data)/2))
	topowx_all_tmean = climateMean(topowx_tmean)
	
	#generate the long term prism averages
	startyear = 1895
	endyear = 2011
	t = np.arange(startyear, endyear)
	prism_long = [[],[],[]]
	days = [31,28,31,30,31,30, 31, 31, 30, 31, 30,31]
	startyear_long = 1895
	endyear = 2011
	var = ['tmin', 'tmax', 'tmean']
	for v in range(len(var)):		
		for year in range(startyear_long, endyear): #PRISM only goes to 2010
			for month in range(1,13):
				fname_prism = 'E:\\PRISM\\%s\\PRISM800m_%s%s_%s.tif' %(var[v], var[v], year, month)
				ds = gdal.Open(fname_prism).ReadAsArray()
				prism_long[v].append(GridData(year, month, days[month-1], var[v], 'PRISM', 'None', ds))			
	
	prism_all_long = []
	prism_all_short = []
	for v in range(len(var)):
		prism_all_long.append(climateMean(prism_long[v]))
		prism_all_short.append(climateMean(prism_long[v][636:]))

	aprism_ppt_short = annualGrid(prism_ppt[636:], p='y')
	prism_ppt_trend = lstFit(aprism_ppt_short)[0]
	
	#write all these for later
	Ny,Nx = np.shape(Ftopowx_trend_mu)
	csize = 0.00833333333
	xll = -112.39583333838
	yul = 46.19583332448
	outpath = 'E:\\TOPOWX\\Analysis\\out\\'
	
	#things to save ppt long and short, topowx tmin, tmax, tmean, prism tmin, tmax, tmean long and short
	
	tiffWrite(prism_all_ppt, Nx, Ny, csize, yul, xll, outpath, "prism_1948_2010_ppt_mean.tif")
	tiffWrite(prism_all_ppt_long, Nx, Ny, csize, yul, xll, outpath, "prism_1895_2010_ppt_mean.tif")
	tiffWrite(topowx_all_tmin, Nx, Ny, csize, yul, xll, outpath, "topowx_1948_2010_tmin.tif")
	tiffWrite(topowx_all_tmax, Nx, Ny, csize, yul, xll, outpath, "topowx_1948_2010_tmax.tif")
	tiffWrite(topowx_all_tmean, Nx, Ny, csize, yul, xll, outpath, "topowx_1948_2010_tmean.tif")
	for v in range(len(var)):
		tiffWrite(prism_all_long[v], Nx, Ny, csize, yul, xll, outpath, "prism_1895_2010_%s_mean.tif"%(var[v]))
		tiffWrite(prism_all_short[v], Nx, Ny, csize, yul, xll, outpath, "prism_1948_2010_%s_mean.tif"%(var[v]))
	tiffWrite(prism_ppt_trend, Nx, Ny, csize, yul, xll, outpath, "prism_1948_2010_ppt_trend.tif")
	
	tiffWrite(prism_ppt_trend*0.0393701*10, Nx, Ny, csize, yul, xll, outpath, "prism_1948_2010_Inch_ppt_trend.tif")
'''
#group all the data together as a single array to generate the table
group_data = [(topowx_all_tmean*(9/5))+32, (topowx_all_tmax*(9/5))+32, (topowx_all_tmin*(9/5))+32, prism_all_ppt*0.0393701, Ftopowx_trend_mu, prism_ppt_trend*0.0393701*10] #Ftopowx is decadal...
#DONE

#group all the masks (area_masks)
#now generate means for all the masks
box_data = []
data_table = []
for i in range(len(group_data)):
	data_row = []
	box_row = []
	for j in range(len(area_masks)):
		data_row.append(np.median(group_data[i][area_masks[j]]))
		box_row.append(group_data[i][area_masks[j]])
	data_table.append(data_row)
	box_data.append(box_row)
box_data = np.array(box_data)
data_table = np.array(data_table)
np.savetxt("climate_summary3.csv", data_table, delimiter=' & ', fmt='%2.2f', newline=' \\\\\n')

#plot by area again...
	#plot!
	my_dpi = 96
	plt.figure(figsize==(800/my_dpi, 800/my_dpi), dpi=my_dpi)
	ax = plt.subplot(111)
	bp = ax.boxplot(box_data[4], notch=0 ,vert=1, whis=1.5, patch_artist=True)
	boxColors = ['black', 'blue', 'lightblue', 'green', 'yellow']
	for patch, color in zip(bp['boxes'], boxColors):
		patch.set_facecolor(color)
	plt.setp(bp['medians'], color='red', lw =2)
	plt.setp(bp['whiskers'], color='black')
	plt.setp(bp['fliers'], color='red', marker='+')
	plt.xticks(np.arange(1,6),a_areas, fontsize = 11)
	plt.ylabel('$^oF/decade$', fontsize=16)
	ax.yaxis.grid(alpha = 0.8)
	plt.savefig('E:\\topowx\\analysis\\out\\box_tmean2.png', bbox_inches='tight', dpi = my_dpi)
	
################################################################################

#consider the means at different elevational bins
	elevation_dtmean = []
	min_ele = np.min(Felevation)
	max_ele = np.max(Felevation)
	dif_ele = max_ele-min_ele
	ele_bins = np.arange(3000,14000,2000)
	ele_labels = []
	for i in range(1,len(ele_bins)):
		elevation_dtmean.append(Ftopowx_trend_mu[(Felevation>ele_bins[i-1]) & (Felevation<ele_bins[i])])
		ele_labels.append('%i-%i'%(ele_bins[i-1],ele_bins[i]-1))
	elevation_dtmean = np.array(elevation_dtmean)
	ax = plt.boxplot(elevation_dtmean)
	
	plt.xticks(np.arange(1,6),ele_labels)
	plt.xlabel('Elevation (ft)', fontsize = 18)
	plt.ylabel(r'TopoWx $\frac{dT_{mean}}{dt}$ ($^oF/decade$)', fontsize = 18)
	
	ens_trend = []
	mask = []
	threshold = []
	binary_mask = []
	
	for i in range(2):
		ens_trend.append((topowx_trend[i]+prism_trend[i])/2)
		mask.append(np.abs(diff_mask[i]))
		threshold.append(np.std(diff_mask[i])*2) #check for values twice the standard deviation
		binary_mask.append(np.where(mask[0]> threshold[0], 0,np.nan))
	
	for i in range(2):
		plt.subplot(2,2,i+1)
		plt.imshow(topowx_trend[i])
		
		
	#look at the elevation trend
	n = np.shape(elevation)[0]*np.shape(elevation)[1]
	trend_series = [np.reshape(ens_trend[0],n),np.reshape(ens_trend[1],n)]
	elev_series = np.reshape(elevation,n)
	
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