# convert probability csv to tiff format
# author: Tony Chang
# date: 9.8.14

import numpy as np
from matplotlib import pyplot as plt
import gdal
from gdalconst import *
import osr

class PGridData(object): #initialize function to construct probability class	
	def __init__(self, year=None, var = None, model = None, rcp = None, data=None):
		self.year = year
		self.var = var
		self.model = model
		self.rcp = rcp
		self.data = data
	def mean(self): #method to get the mean of the domain
		return(np.mean(self.data))

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

def Tiffwrite(data,Nx,Ny,cellsize,yul,xll, path, name):   
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
	
def writetifdata(data, list, workspace,gcm = 'n'):
	mdata = monthlydatalist(data)
	head = Headerextract(gcm)
	for i in range(len(mdata)):
		for j in range(12):
			path = workspace + '\\' 
			name = list[i] + str(j+1) +'.tif'
			Tiffwrite(mdata[i][j], head['ncols'], head['nrows'], head['csize'],head['yul'], head['xul'], path, name)
			
# Get the Data
#===================================================================================================================
#===================================================================================================================
workspace = "E:\\wbp_model\\prob_output_04172014\\"
models = ['CESM1-BGC']
rcps = [85]
syear = 2010
eyear = 2099
n = eyear-syear
nrows, ncols = 471,504 #dimensions of the grids for GYE

yprobs = []

for model in models:
	for rcp in range(len(rcps)): #next parameter is the rcp
		filename = workspace + model + '\\' + model + '_' + str(rcps[rcp]) + '_' + str(syear) + '-' + str(eyear) + '_probs.csv'
		data = np.genfromtxt(filename, delimiter = ',', names = True)
		for i in range(n):
			probs =  np.reshape(data[str(syear+i)],(nrows, ncols))
			yprobs.append(PGridData(i+syear, 'prob_pres', model, rcps[rcp], probs))


# generate the ensemble averages and output as a tiff

h = Headerextract()
workspace = "E:\\earth\\WBP_projections\\"

for y in range(eyear-syear):
	name = "WBP_prob_%s_%i.tif" %(models[0],y+syear)
	Tiffwrite(yprobs[y].data, h['ncols'], h['nrows'], h['csize'], h['yul'], h['xul'], workspace, name)
	
# output the tiff file

