# -*- coding: utf-8 -*-
"""
Created on Fri Feb 01 15:07:55 2013

@author: tony.chang
Climatic water deficit equations from 
Lutz, J.A., van Wagtendonk, J.W., and Franklin, J.F. Climatic water deficit, 
tree species, and climate change in Yosemite National Park. 2010. 
Journal of Biogeography

"""
import math as math
import numpy as np
import gdal 
from gdalconst import *
import osr 
import matplotlib.pyplot as plt
from matplotlib import mpl
from pylab import *
import time
import matplotlib.animation as animation
import csv
from collections import OrderedDict

#Initialization
"""Required variables to have for calculations
    Ta=         #mean monthly temperature
    Pm=         #mean monthly precipitation
    slope=      #slope of grid cell
    aspect=     #aspect of grid cell
    lat=        #latitude of grid cell
    sd=         #solar declination angle at noon on the 15th day of the month
    soilm=      #soil moisture values

Boundary edges for site extent are WGS72
ymax = 40.0041666603
xmin = -91.0041666803
xmax = -77.0041666859
ymin = 31.0041666639
"""
#define PRISM data class
class PRISMData(object):
    #initialize function to construct PRISMdata class
    def __init__(self, year=None, month=None, data=None):
        self.year = year
        self.month = month
        if month == 14:         #month 14 in PRISM data represents the mean of the years
            self.season = "ALL"
        elif (month < 3 or month == 12):
            self.season = "Win"
        elif month < 6:
            self.season = "Spr"
        elif month < 9:
            self.season = "Sum"
        else:
            self.season = "Fal"
        self.data = data

class GCMData(object):
    #initialize function to construct PRISMdata class
    def __init__(self, year=None, month=None, data=None):
        self.year = year
        self.month = month
        if month == 14:         #month 14 in PRISM data represents the mean of the years
            self.season = "ALL"
        elif (month < 3 or month == 12):
            self.season = "Win"
        elif month < 6:
            self.season = "Spr"
        elif month < 9:
            self.season = "Sum"
        else:
            self.season = "Fal"
        self.data = data
		
class WBmodel(object):
    def __init__(self, year=None, month=None, Ta=None, Fm=None, rain=None, snow=None, pack=None):
        self.year = year
        self.month = month
        if month == 14:         #month 14 in data represents the mean of the years
            self.season = "ALL"
        elif (month < 3 or month == 12):
            self.season = "Win"
        elif month < 6:
            self.season = "Spr"
        elif month < 9:
            self.season = "Sum"
        else:
            self.season = "Fal"
        self.Ta = Ta
        self.Fm = Fm
        self.rain = rain
        self.snow = snow
        self.pack = pack
        
"""============================================================================
====================PRISM extraction methods===================================    
============================================================================="""

#Header extract function
def Headerextract():
    #takes arbitrary PRISM dataset and extracts the header parameters
    filename = "D:\\CHANG\\GIS_Data\\SE_data\\TIFF\\800m_tiff\\tmin\\PRISM800m_tmin1980_1.tif" 
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
"============================================================================="

#PRISM data extract function    
def PRISMextract(BeginYear,EndYear,var):        
    Pdata = [] #List to store all PRISMData object
    workspace = "D:\\CHANG\\GIS_Data\\SE_data\\TIFF\\800m_tiff\\"+ var +"\\"    
    for searchyear in range(BeginYear, EndYear+1): #looping through years of interest
        for filenum in range(1,13): #does not consider the annual mean filenum (#14)
            filename = workspace + "PRISM800m_" + var + str(searchyear) + "_" + str(filenum) + ".tif"
            readfile =  gdal.Open(filename)
            data = np.array(readfile.GetRasterBand(1).ReadAsArray())
            x = PRISMData(searchyear,filenum,data) #Create instance of PRISMData object
            Pdata.append(x) 
            readfile = None #close file
    return(Pdata)
	
"""=============================================================================
========================GCM Extraction=========================================
============================================================================"""
from netCDF4 import Dataset
'''GCM extents are different!

GCM
xmin = -112.42916667169999
xmax = -108.27916667336001
ymin = 42.262499992719995
ymax = 46.179166657819998

PRISM
xmin = -112.438
xmax = -108.271
ymin = 42.262
ymax = 46.187

This is cut one row short on top and on the right
'''
def GCMextract(beginyear,endyear,var,rcp):
	if (var == 'ppt'):
		v = 'pr'
	elif (var =='tmin'):
		v = 'tasmin'
	elif (var == 'tmax'):
		v = 'tasmax'
	workspace = "E:\\GCM\\cesm1-bgc\\cesm1-bgc\\"
	GCMdata = []
	for cyear in range(beginyear,endyear+1):
		filename = workspace + "BCSD_0.008deg_"+v+"_Amon_CESM1-BGC_rcp"+str(rcp)+"_r1i1p1_"+str(cyear)+"01-"+str(cyear)+"12.nc"
		rootgrp = Dataset(filename,'r',format='NETCDF4')
		for month in range(0,12):
			data = rootgrp.variables[v][month]
			if (v=='pr'):
				multiplier = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
				data = multiplier[month]*data[::-1] #need to mirror the matrix, convert from mm/day to mm/month
			else:
				data = data[::-1] #need to mirror the matrix
			x = GCMData(cyear,month,data)
			GCMdata.append(x)
		rootgrp.close()
	return(GCMdata)
		
"""============================================================================
========================Utilities==============================================
============================================================================"""
#Average monthly temperature function
def MeanTemp(BeginYear,EndYear):
    Tmax = PRISMextract(BeginYear,EndYear,'tmax')
    Tmin = PRISMextract(BeginYear,EndYear,'tmin')    
    Tmean = []
    for i in range(len(Tmax)):
        storearray = PRISMData(Tmax[i].year, Tmax[i].month, ((Tmax[i].data + Tmin[i].data)/2)) #Definition of Tmean from Daly 2012
        #storearray = PRISMData(Tmax[i].year, Tmax[i].month, ((0.606*Tmax[i].data/100.) + (0.394*Tmin[i].data/100.))) #definition of Tave from Thornton et al 1997 and Running et al 1987 (specific to daily data)       
        Tmean.append(storearray)
    return(Tmean)

# Vapor pressure calculation function
def VapPD(Tdmean, Tmean): #vapor pressure calculations from Campbell and Norman 1998, adapted from Weiss et al 2012
    a = 0.611 #kPa
    b = 17.502
    c = 240.97 # deg C
    VPsat = []
    VPact = []
    VPDmean = []
    for i in range(len(Tmean)):    
        VPsatstorearray = PRISMData(Tmean[i].year, Tmean[i].month, a*np.exp(((b*Tmean[i].data)/(Tmean[i].data+c))))
        VPactstorearray = PRISMData(Tmean[i].year, Tmean[i].month, a*np.exp(((b*(Tdmean[i].data))/((Tdmean[i].data)+c))))
        VPDmeanstorearray = PRISMData(Tmean[i].year, Tmean[i].month, VPsatstorearray.data-VPactstorearray.data)         
        VPsat.append(VPsatstorearray)
        VPact.append(VPactstorearray)
        VPDmean.append(VPDmeanstorearray)
    return(VPsat,VPact,VPDmean)
   
def GDD5(BeginYear, EndYear):
	#Growing degree day calculations based on Sork et al 2010. 
	dayspmonth = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
	dthresh = 5 #degree C defining degree day 
	tmax = PRISMextract(BeginYear, EndYear, 'tmax')
	tmin = PRISMextract(BeginYear, EndYear, 'tmin')
	nrows,ncols = np.shape(tmax[0].data)
	GDD5array = []
	annualgdd5 = np.zeros((nrows,ncols))
	for i in range(len(tmax)):
		if (i==0): #first iteration
			m = tmax[i].month -1
			D = np.ones((nrows,ncols))*dayspmonth[m]
			# adjust day number for regions where Tmin < 5 and Tmax >5
			testarray = (tmax[i].data>dthresh) & (tmin[i].data<dthresh)
			D[testarray] = dayspmonth[m] *(tmax[i].data[testarray] - dthresh)/ (tmax[i].data[testarray]-tmin[i].data[testarray])
			annualgdd5 += (((tmin[i].data + tmax[i].data)/2)-5)*D	
		elif (tmax[i].year == tmax[i-1].year): #while the year is the same as the previous iteration sum the values of the degree days
			m = tmax[i].month -1
			D = np.ones((nrows,ncols))*dayspmonth[m]
			# adjust day number for regions where Tmin < 5 and Tmax >5
			testarray = (tmax[i].data>dthresh) & (tmin[i].data<dthresh)
			D[testarray] = dayspmonth[m] *(tmax[i].data[testarray] - dthresh)/ (tmax[i].data[testarray]-tmin[i].data[testarray])
			annualgdd5 += (((tmin[i].data + tmax[i].data)/2)-5)*D	
		else: #if new year, append the annualggd to GDD5arary and start a new annualggd5
			GDD5array.append(annualgdd5)
			annualgdd5 = np.zeros((nrows,ncols))
			m = tmax[i].month -1
			D = np.ones((nrows,ncols))*dayspmonth[m]
			# adjust day number for regions where Tmin < 5 and Tmax >5
			testarray = (tmax[i].data>dthresh) & (tmin[i].data<dthresh)
			D[testarray] = dayspmonth[m] *(tmax[i].data[testarray] - dthresh)/ (tmax[i].data[testarray]-tmin[i].data[testarray])
			annualgdd5 += (((tmin[i].data + tmax[i].data)/2)-5)*D
	return(np.array(GDD5array))
	
#coordinate locator function based on data matrix indices
def Cretrieve(xindex,yindex):
    header = Headerextract()
    xmin = header['xul']    
    ymin = header['yul']
    csize = header['cize']
    lat = ymin-yindex*csize
    lon = xmin+xindex*csize
    coor = [lat,lon]
    return(coor)

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

def Prec(BeginYear,EndYear):
    ppt = PRISMextract(BeginYear,EndYear,'ppt')
    for i in range(len(ppt)):
        ppt[i].data = ppt[i].data
    return(ppt)

def Topoextract():
    aspectPath = "D:\\CHANG\\GIS_Data\\SE_data\\TIFF\\SE_asp800m.tif"
    slopePath = "D:\\CHANG\\GIS_Data\\SE_data\\TIFF\\SE_slope800m.tif"
    elevPath = "D:\\CHANG\\GIS_Data\\SE_data\\TIFF\\SE_DEM800m.tif"   
    ds = gdal.Open(aspectPath)
    aspect = np.array(ds.GetRasterBand(1).ReadAsArray())
    #aspect = ds.GetRasterBand(1).ReadAsArray()
    ds = gdal.Open(slopePath)
    slope = np.array(ds.GetRasterBand(1).ReadAsArray())
    #slope = ds.GetRasterBand(1).ReadAsArray()
    ds = gdal.Open(elevPath)
    elev = np.array(ds.GetRasterBand(1).ReadAsArray())
    #elev = ds.GetRasterBand(1).ReadAsArray()
    ds = None #close files
    return(aspect,slope,elev)    
    #return(np.array(aspect),np.array(slope),np.array(elev))

"""============================================================================
====================Water Balance Models=======================================    
============================================================================="""

def Meltfactor(Ta):  
    Fm = Ta
    Fm[Fm<0] = 0 #first test if Ta <= 0
    Fm = Fm*0.167    
    Fm[Fm>1.002] = 1 #second test if Ta >= 6*0.167, set equal to 1 
    return (Fm)
    
def Meltdata_create(Ta):
    Fm_list = []
    for i in range(len(Ta)):
        Fm = Meltfactor(Ta[i].data)
        Fm_list.append(Fm)
    return(Fm_list)
    
def Pack(Fm, Pm):    
    pack = []    
    for i in range(len(Fm)):
        if (i == 0): 
            monthpack = ((1-Fm[i])**2 * Pm[i].data) #initial pack condition assumed to be 0
        else:
            monthpack=(((1-Fm[i])**2 * Pm[i].data) + ((1-Fm[i]) *pack[i-1]))
        pack.append(monthpack)
    return(pack)
    
def Waterinput(Fm, Pm, pack):
    wm_list = []
    melt = []
    rain = []    
    for i in range(len(Fm)):            
        rain_m = Fm[i]*Pm[i].data #portion of rain water that is not snow
        snow_m = (1-Fm[i])*Pm[i].data
        if i == 0:
            melt_m = Fm[i] * (snow_m)  #pack assumed to be 0 at first month
        else:
            melt_m = Fm[i] * (snow_m + pack[i-1])
        wm = rain_m + melt_m
        melt.append(melt_m)
        rain.append(rain_m)
        wm_list.append(wm)
    return(wm_list,melt,rain)
    
def Aspectf(aspect): #takes in aspect in degrees
    af = np.ma.masked_array(aspect, aspect==-1)
    af.fill_value = np.nan     #maybe adjust this so that all aspect values of -1 do not get calculated
    af = af.filled()
    af = abs(math.pi-abs((np.radians(af))-(math.pi*5/4)))    
    return(af) #returns the folded aspect in radians format

def Solardec():
    #calculates the solar declination angle for a given latitude and month 
    #We use the solar declination angle at noon on the 15th of the given month    
    daypmon = np.array([31,28,31,30,31,30,31,31,30,31,30,31]) #number of days per month, not considering leap year
    D = daypmon-15
    ffthday = []
    for i in range(len(daypmon)):
        if i == 0:
            ffthday.append(daypmon[i]-D[i])
        else:
            ffthday.append(ffthday[i-1]+D[i-1]+daypmon[i]-D[i])
    ffthday = np.array(ffthday)
    sd = -23.45*(math.pi/180)* np.cos((2*math.pi)*(ffthday+10)/365) 
    return(sd)

def Daylength(sd,lat):  
    av = 0.2618 #angular velocity of the Earth's rotation (rad/hr)        
    lat = np.radians(lat) #convert to radians    
    dl = []    
    for i in range(len(sd)):
        dl.append((2*np.arccos(-np.tan(sd[i])*np.tan(lat)))/av)
    dl = np.array(dl) #day length
    return(dl)
    
def HeatLI(slope,aspect):    
    af = (Aspectf(aspect))
    slope = np.radians(slope) #convert to radians
    lat,lon = Gridcreate()
    lat = np.radians(lat) #convert to radians
    HL = 0.339 + 0.808*(np.cos(lat)*np.cos(slope)) - 0.196*(np.sin(lat)*np.sin(slope)) - 0.482*(np.cos(af)*np.sin(slope))
    nanIndex = np.isnan(HL) #locate all the nan values (no aspect)
    fillnan = HL[nanIndex] #index of the nan values
    fillnan = 0.339 + 0.808*(np.cos(lat[nanIndex])*np.cos(slope[nanIndex])) - 0.196*(np.sin(lat[nanIndex])*np.sin(slope[nanIndex])) #replace nan values with HL equation without aspect load
    HL[nanIndex] = fillnan #insert values into nan elements
    return(HL)
    
def PETcalc(Ta,month,hl):    #This is based on the Thornthwaite equation
    ea = 0.611*(np.exp((17.27*Ta)/(Ta+237.3))) #saturation vapour pressure
    dayspmonth = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    days = np.ones(np.shape(Ta))*dayspmonth[month-1] #create grid of shape(Ta) filled with number of days of given month   
    sd = Solardec()
    lat,lon = Gridcreate()
    #dl = Daylength(sd,lat)
    dl = (Daylength(sd,lat)) #divide by 24 to convert from hours to days <--check this!
    PET = 29.8*days*dl[month-1]*hl*(ea/(Ta+273.2))
    return(PET)

def PET(Ta,HL):
    p=[]
    for i in range(len(Ta)):
        p.append(PETcalc(Ta[i].data,Ta[i].month,HL))
    return(p)

def DeltaSoilCalc(soilmax,Wm,pet):#calculate the soil moisture
	soilm = []
	AETarray = []
	for i in range(len(pet)):   
		if (i ==0):
			soilm.append(np.ones(np.shape(pet[i]))*1e-4) #start soil moisture at zero
			AETarray.append(pet[i])
		else:
			#calculate 2 seperate cases for AET, based on the relationship of Wm and PET
			#case 1 if Wm>PET
			tempAET1 = pet[i]
			tempsoilmoisture1 = np.minimum(Wm[i]-pet[i]+soilm[i-1], soilmax)
			#case 2 if Wm<=PET
			dsoil = soilm[i-1] * (1- np.exp((Wm[i]- pet[i])/soilmax))
			tempAET2 = Wm[i]+ dsoil
			tempsoilmoisture2 = np.minimum(Wm[i] - tempAET2 +soilm[i-1], soilmax) #water for the month is the water that came in, what came out, plus what was before          
			#now run the conditions on the arrays
			AET = np.where(Wm[i]>pet[i], tempAET1, tempAET2)
			soilmoisture = np.where(Wm[i]>pet[i], tempsoilmoisture1, tempsoilmoisture2)
			soilm.append(soilmoisture) #the culmulative soil moisture for the time step
			AETarray.append(AET)        
	return(soilm,AETarray)
	

def SoilWHC(): #defines the soil water holding capacity, currently using STATSGO dataset with depth at 150cm
	header = Headerextract()
	soilPath = "D:\\CHANG\\GIS_Data\\SE_Data\\TIFF\\awc800m.tif" 
	ds = gdal.Open(soilPath)
	uvalue = 10 #AWC values are in centimeters, multiply by this to convert to mm
	whc = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype='f8') 
	whc[np.where(whc==100)] = 150 #this represents lakes and basins
	whc[np.where(whc==0)] = 1e-4 #this represents impermeable surfaces
	whc = whc*uvalue
	#uvalue= 200 #for uniform soil water holding capacity
	#whc = uvalue * np.ones([header['nrows'], header['ncols']]) #for the moment creates a uniform soil water holding capacity of 100mm
	return(whc)



"""=============================================================================
========================Data Analysis===========================================
============================================================================="""
def PGridmean(data,startyear,endyear):
	n,m = data[0].data.shape
	psum = np.zeros((n,m))
	pmean=[]
	gridmeans = []
	counter = 0
	for i in range(12):
		pmean.append(psum) #12 instances of np.zeros((n,m))
	for j in range(len(data)):#loop through all the data 
		if(data[j].year >= startyear and data[j].year <=endyear):
			pmean[data[j].month-1] = pmean[data[j].month-1] + data[j].data
			if (data[j].month-1 ==0):
				counter +=1
	for k in range(12):
		pmean[k] = pmean[k]/counter #calculate the average for the time period of the data
		gridmeans.append(pmean[k].mean())
	return (pmean,np.array(gridmeans))

def WBGridmean(data,burnin):
	n,m = np.shape(data[0])
	psum = np.zeros((12,n,m))
	gridmeans = []
	counter = 0
	for j in range(burnin*12,len(data),12):
		for k in range(0,12):
			psum[k] += data[j+k]
		counter +=1
	psum = psum/counter
	for z in range(12):
		gridmeans.append((psum[z]).mean())
	return(psum,np.array(gridmeans))
		
def Periodmean(data,burnin): #burnin is the number of months to skip to allow soils to saturate with water
	n,m = np.shape(data[0])
	psum = np.zeros((n,m))
	df = len(data)
	counter = 0
	for i in range(burnin-1,df):
		psum += data[i]
		counter +=1
	return (psum/counter)
	
def Dictwrite(outdata):
	#write data to file
	outdata = OrderedDict(sorted(outdata.items()))
	header = list(outdata.keys())
	values = list(outdata.values())
	f = open('D:\\chang\\python_scripts\\output\\samplesummary.csv','w')
	for hi in range(len(header)):
		if (hi == len(header)-1):
			f.write(header[hi]+'\n') #end header and write new line
		else:
			f.write(header[hi]+',')
	for vi in range(len(values[0])):
		for vj in range(len(values)):
			if(vj==len(values)-1):
				f.write(str(values[vj][vi])+'\n')
			else:
				f.write(str(values[vj][vi])+',') #write row
	f.close()
	return()
	
"""===============================================================================
=======================Write Functions============================================
==============================================================================="""
def PRISMtiffwrite(data,var,name,Nx,Ny,cellsize,yul,xll,nbands):    
    fileformat = "GTiff"
    driver = gdal.GetDriverByName(fileformat)
    geotransform = [xll, cellsize,0.0,yul, 0.0, -cellsize]
    srs = osr.SpatialReference()
    #srs.ImportFromEPSG(4322) #4322 coordinate ref# for World Geodetic System 1972
    #writename = "C:\CHANG\PRISM\PRISM_Analysis\us_" + var + "_" + str(BeginYear) + "_" + str(EndYear) + "_gradients.tif"
    writename = "D:\\CHANG\\climate_models\\us_prism_800m\\uncompressed\\800m_tiff\\"+ var + "\\" + name
    outDs = driver.Create(writename, Nx, Ny, nbands, gdal.GDT_Float32)
    outDs.SetGeoTransform(geotransform)
    srs.SetWellKnownGeogCS("WGS72")
    outDs.SetProjection(srs.ExportToWkt())
    for band in range(nbands):
        outBand = outDs.GetRasterBand(band+1)
        #outBand = outDs.GetRasterBand(band+1)
        outBand.SetNoDataValue(-9999)
        #outBand.WriteArray(data[band],0,0)
        outBand.WriteArray(data,0,0)
    outDs = None
    return(print(writename + "filebuilt!\n"))

def Tiffwrite(data,var,name,Nx,Ny,cellsize,yul,xll,nbands):   
    fileformat = "GTiff"
    driver = gdal.GetDriverByName(fileformat)
    geotransform = [xll, cellsize,0.0,yul, 0.0, -cellsize]
    srs = osr.SpatialReference()
    #writename = "C:\CHANG\PRISM\PRISM_Analysis\us_" + var + "_" + str(BeginYear) + "_" + str(EndYear) + "_gradients.tif"
    writename = "D:\\CHANG\\climate_models\\us_prism_800m\\uncompressed\\800m_tiff\\30yearmeans\\"+ var + "_"+ name +".tif"
    outDs = driver.Create(writename, Nx, Ny, nbands, gdal.GDT_Float32)
    outDs.SetGeoTransform(geotransform)
    srs.SetWellKnownGeogCS("WGS72")
    outDs.SetProjection(srs.ExportToWkt())
    for band in range(nbands):
        outBand = outDs.GetRasterBand(band+1)
        outBand.SetNoDataValue(-9999)
        outBand.WriteArray(data,0,0)
    outDs = None
    return(print(writename + "filebuilt!\n"))

def Tiffwrite2(data, name = 'Temp'):
	head = Headerextract()
	xll = head['xul']
	yul = head['yul']
	cellsize = head['csize']
	nbands = head['bands']
	Nx = head['ncols']
	Ny = head['nrows']
	fileformat = "GTiff"
	driver = gdal.GetDriverByName(fileformat)
	geotransform = [xll, cellsize,0.0,yul, 0.0, -cellsize]
	srs = osr.SpatialReference()
	writename = "D:\\Chang\\GIS_data\\se_data\\Climate\\" + name + ".tif"
	#writename = "C:\CHANG\PRISM\PRISM_Analysis\us_" + var + "_" + str(BeginYear) + "_" + str(EndYear) + "_gradients.tif"
	#writename = "D:\\CHANG\\Python_scripts\\output\\"+ name +".tif"
	#outDs = driver.Create(writename, Nx, Ny, nbands, gdal.GDT_Float32)
	outDs = driver.Create(writename, Nx, Ny, nbands, gdal.GDT_Float32)
	outDs.SetGeoTransform(geotransform)
	srs.SetWellKnownGeogCS("WGS72")
	outDs.SetProjection(srs.ExportToWkt())
	for band in range(nbands):
		outBand = outDs.GetRasterBand(band+1)
		outBand.SetNoDataValue(-9999)
		outBand.WriteArray(data,0,0)
	outDs = None
	return(print(writename + " filebuilt! \n"))

def Tiffwrite3(data,var,name,Nx,Ny,cellsize,yul,xll):   
	fileformat = "GTiff"
	nbands = 1
	driver = gdal.GetDriverByName(fileformat)
	geotransform = [xll, cellsize,0.0,yul, 0.0, -cellsize]
	srs = osr.SpatialReference()
	writename = "D:\\CHANG\\python_scripts\\output\\climate_tiff\\projections\\"+ var + "_"+ name +".tif"
	outDs = driver.Create(writename, Nx, Ny, nbands, gdal.GDT_Float32)
	outDs.SetGeoTransform(geotransform)
	srs.SetWellKnownGeogCS("WGS72")
	outDs.SetProjection(srs.ExportToWkt())
	for band in range(nbands):
		outBand = outDs.GetRasterBand(band+1)
		outBand.SetNoDataValue(-9999)
		outBand.WriteArray(data,0,0)
	outDs = None
	return(print(writename + "filebuilt!\n"))
	
def Writesample(sampledata,fname,name='sample'):
	head = ','.join(fname)
	np.savetxt("D:\\Chang\\python_scripts\\output\\" + name+".csv", sampledata, delimiter=',', header = head)
	return(head)

"""=========================================================================
=========================Population data functions==========================
========================================================================="""
def WBPextract(): #extracts the samples
    filename = "D:\\Chang\\gis_data\\fia\\WBPdatawgs72.csv"
    dt = [('FID', 'f8'),('CN', 'f8'), ('MYEAR', 'i4'),('LAT', 'f8'), ('LON', 'f8'), ('ELEV', 'f8'),('PR', 'i1'), ('SEEDLING', 'f8'), ('GR', 'f8'), ('REP', 'f8'), ('MORT', 'f8'), ('DEM', 'f8'),('E_DIFF', 'f8'), ('px','f8'), ('py', 'f8')]
    data = np.genfromtxt(filename, dtype = dt, delimiter = ',',names=True)
    return(data)    

def WLISprep():
	filename = "D:\\chang\\phd_material\\wbp_project\\data\\wlis_rockies_filtered2.csv"
	dt =[('SurveyYear', '<f8'), ('Species', '|S8'), ('WPBRPresent', '|S8'), ('WLP_M_NE', '|S8'), ('OtherSpecies_M_NE', '|S8'), ('StateProv', '|S8'),('Latitude', '<f8'), ('Longitude', '<f8'), ('PerMortAll', '|S8'), ('PerMortBR_', '|S8'), ('Regenerati', '|S8'), ('CN', '<f8'), ('Elevation', '<f8')]
	wlis = np.genfromtxt(filename, delimiter = ',', names = True, dtype=dt)
	mortindex = np.where(wlis['PerMortAll'] == b' Y')
	nanindex = np.where(wlis['PerMortAll'] == b' NE')
	mortlist = wlis['PerMortAll']
	mortlist[mortindex] =1
	mortlist[nanindex] = np.nan
	reg = wlis['Regenerati']
	reg[np.where(reg == b' Y')] = 1
	reg[np.where(reg == b' N')] = 0
	reg[np.where(reg == b' NE')] = np.nan
	reg = reg.astype('f8')
	mortlist = mortlist.astype('f8')
	wlisdata = np.array([wlis['CN'],wlis['Latitude'], wlis['Longitude'], wlis['Elevation'], np.ones(len(reg)),mortlist,reg,np.ones(len(reg))])
	wlisdata = wlisdata.T[np.where(wlisdata.T[:,3]!=-9999)] #remove the null elevation values
	#wfname = np.array(['CN', 'Lat', 'Long', 'Elev', 'Pres', 'Mort', 'Seedl', 'Reprod'])
	wfname = {'CN':0, 'Lat':1, 'Lon':2, 'Elev':3, 'Pres':4, 'Mort':5, 'Seedl':6, 'Reprod':7} #index key
	return(wlisdata, wfname)

def Datamerge(wlisdata,wfname, sdata): #connects the WLIS dataset with the FIA dataset
	max_el_diff = 1000 #specify the maximum difference in elevation that is acceptable of FIA to USGS DEM
	data = sdata[np.where(sdata['ELEV_DIFF']<max_el_diff)] #filtered dataset
	fia = np.array([data['CN'], data['POINT_Y'], data['POINT_X'], data['ELEV'], data['PRESENT'], data['MORT'], data['SEEDLING'], data['REPROD']])
	merged_data = np.vstack((fia.T, wlisdata)) #merge datasets
	#now find the data that relates to points with mortality
	#mi = np.where(merged_data[:,wfname['Mort']]!=0)
	#merged_data[:,wfname['Pres']][mi] = 2 #assign a new class of presence
	mdata = {'CN':merged_data[:,0], 'lat':merged_data[:,1], 'lon':merged_data[:,2], 'elev':merged_data[:,3], 'pres':merged_data[:,4], 'mort':merged_data[:,5],'seedl':merged_data[:,6], 'reprod':merged_data[:,7]}
	return(mdata)

def Sampleindexextract(mdata): #exports data features that match sample spatial coordinates
	y,x=Gridcreate() #create the grid from the data extent
	x = x[0]
	y = y[:,0]
	n = len(mdata['CN'])
	sx = mdata['lon']
	sy = mdata['lat']
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
	return(indexarray)

def Dataextract(mdata,data):
	sind = Sampleindexextract(mdata)
	datavaluelist = []
	for i in range(len(sind['xi'])):
		if (np.isnan(sind['xi'][i])): #if the sample reference to data is nan
			datavaluelist.append(np.nan)
		else:
			row = int(sind['yi'][i])
			col = int(sind['xi'][i])
			datavaluelist.append(data[row][col])
	return(np.array(datavaluelist))
	
def Datacompile(mdata, Cgmean,Cbar,Wmean): #adds features to the sample set and outputs csv
	tmonhot = np.argsort(Cbar['tmax'])[:8:-1] #3 hottest months
	tmoncold = np.argsort(Cbar['tmin'])[:3] #3 coldest months
	pptwet = np.argsort(Cbar['ppt'])[:8:-1] #3 wettest months
	pptdry = np.argsort(Cbar['ppt'])[:3] #3 driest months
	outdata = Sampleindexextract(mdata)
	outdata['lat'] = mdata['lat']
	outdata['long'] = mdata['lon']
	outdata['elev'] = mdata['elev']
	outdata['presence'] = mdata['pres']
	outdata['mort'] = mdata['mort']
	outdata['seedl'] = mdata['seedl']
	outdata['reprod'] = mdata['reprod']
	outdata['CN'] = mdata['CN']
	for i in range(3):
		outdata['Tmax'+str(i+1)+'_mon'+str(tmonhot[i]+1)] = Dataextract(mdata,Cgmean['tmax'][tmonhot[i]])
		outdata['Tmin'+str(i+1)+'_mon'+str(tmoncold[i]+1)] = Dataextract(mdata,Cgmean['tmin'][tmoncold[i]])
		outdata['Pwet'+str(i+1)+'_mon'+str(pptwet[i]+1)] = Dataextract(mdata,Cgmean['ppt'][pptwet[i]])
		outdata['Pdry'+str(i+1)+'_mon'+str(pptdry[i]+1)] = Dataextract(mdata,Cgmean['ppt'][pptdry[i]])
	outdata['PET'] = Dataextract(mdata,Wmean['pet'])
	outdata['AET'] = Dataextract(mdata,Wmean['aet'])
	outdata['Pack'] = Dataextract(mdata,Wmean['pack'])
	outdata['Soil_m'] = Dataextract(mdata,Wmean['soilm'])
	return(outdata)

def Sampleprep(out,out2):
	#takes in the outputs for 2 time periods and returns the sample in numpy array format for ML fitting
	sampledata = []
	fname = []
	ind = ~np.isnan(out['xi']) #find all the nan values to remove
	#ind2 = np.where(out['elev_diff_from_dem'][ind]<1000) #find all the samples that are less than 1000m from the actual elevation
	sampledata.append(out['CN'][ind])
	fname.append('CN')
	sampledata.append(out['lat'][ind])
	fname.append('lat')
	sampledata.append(out['long'][ind])
	fname.append('long')
	sampledata.append(out['elev'][ind])
	fname.append('elev')
	#sampledata.append(out['elev_diff_from_dem'][ind])
	#fname.append('elev_dif')
	sampledata.append(out['presence'][ind])
	fname.append('presence')
	#sampledata.append(out['growth'][ind])
	#fname.append('growth')
	sampledata.append(out['mort'][ind])
	fname.append('mort')
	sampledata.append(out['seedl'][ind])
	fname.append('seedl')
	sampledata.append(out['reprod'][ind])
	fname.append('reprod')
	k = sorted(out.keys())
	del k[1]
	k.insert(3,k.pop(9)) #move soil_m next to the other water balance features
	for i in range(16):
			sampledata.append(out[k[i]][ind])
			fname.append('1950_1980_' + k[i])
	k2 = sorted(out2.keys())  #remove this section if only one time period is of interest
	del k2[1]
	k2.insert(3,k2.pop(9))
	for i in range(16):
			sampledata.append(out2[k2[i]][ind])
			fname.append('1980_2010_' + k2[i])
	sampledata = np.array(sampledata)
	return(sampledata.T, fname)

"""============================================================================
=============================MODEL DATA PREP==========================================
============================================================================="""
	
def RFdataprep(sampledata,fname, testdata,testfname,testdata2,testfname2):
	y = sampledata.T[4] #extract all the presence and life history data
	ylabel = fname[4]
	x = sampledata.T[8:]
	xfname = fname[8:]
	testset = []
	testfeat = []
	for i in range(1,len(testfname)):
		testset.append(testdata.T[i])
		testfeat.append('1950_1980_' + testfname[i])
	for i in range(4,len(testfname2)):
		testset.append(testdata2.T[i])
		testfeat.append('1980_2010_' + testfname2[i])
	testset = np.array(testset).T
	return(y,ylabel,x.T, xfname, testset,testfeat)	
	
def Testprep(Cgmean,Cbar,Wmean,topo): #generates grid of extent and extracts data for testing model
	sy,sx = Gridcreate()
	nr,nc = np.shape(sy)
	newsize = nr*nc
	sid = arange(newsize) + 1
	testdata = [sid]  #create single column for analysis
	testdata.append(sx.reshape(1,newsize)[0])
	testdata.append(sy.reshape(1,newsize)[0])
	testdata.append(topo['elev'].reshape(1,newsize)[0])
	testdata.append(Wmean['aet'].reshape(1,newsize)[0])
	testdata.append(Wmean['pet'].reshape(1,newsize)[0])
	testdata.append(Wmean['pack'].reshape(1,newsize)[0])
	testdata.append(Wmean['soilm'].reshape(1,newsize)[0])
	fname = ['sid', 'lon','lat','elev','AET','PET','pack','soil_m'] #feature names
	tmonhot = np.argsort(Cbar['tmax'])[:8:-1] #3 hottest months
	tmoncold = np.argsort(Cbar['tmin'])[:3] #3 coldest months
	pptwet = np.argsort(Cbar['ppt'])[:8:-1] #3 wettest months
	pptdry = np.argsort(Cbar['ppt'])[:3] #3 driest months
	for i in range(3):													#Ugly coding, but for formmating sake....
		testdata.append(Cgmean['ppt'][pptdry[i]].reshape(1,newsize)[0])
		fname.append('Pdry'+str(i+1)+'_mon'+str(pptdry[i]+1))
	for i in range(3):
		testdata.append(Cgmean['ppt'][pptwet[i]].reshape(1,newsize)[0])
		fname.append('Pwet'+str(i+1)+'_mon'+str(pptwet[i]+1))
	for i in range(3):
		testdata.append(Cgmean['tmax'][tmonhot[i]].reshape(1,newsize)[0])
		fname.append('Tmax'+str(i+1)+'_mon'+str(tmonhot[i]+1))
	for i in range(3):
		testdata.append(Cgmean['tmin'][tmoncold[i]].reshape(1,newsize)[0])
		fname.append('Tmin'+str(i+1)+'_mon'+str(tmoncold[i]+1))
	testdata = np.array(testdata)
	fname = np.array(fname)
	return(testdata.T,fname)				

"""=============================================================================
========================DATA INITIALIZATION COMMANDS============================
============================================================================"""
def Climatesummary(BeginYear,EndYear, burnin):
	Start = BeginYear+burnin #we want to consider the year after the burnin
	Pm = Prec(Start,EndYear)
	Tmax = PRISMextract(Start,EndYear,'tmax')
	Tmin = PRISMextract(Start,EndYear,'tmin')  
	Tmaxmean, Tmaxbar = PGridmean(Tmax,Start,EndYear) 	#to do, make a function for this to gather all the climate summaries
	Tminmean, Tminbar = PGridmean(Tmin,Start,EndYear) 	#
	Pptmean, Pptbar = PGridmean(Pm,Start,EndYear) 		#
	Cgridmean = {'tmax':Tmaxmean,'tmin':Tminmean,'ppt':Pptmean}
	Cbar = {'tmax':Tmaxbar,'tmin':Tminbar,'ppt':Pptbar}
	return(Cgridmean,Cbar)

def Watersummary(pet,pack,soilm,aet, burnin):
	burnin = burnin*12 #multiply the burnin by 12 months
	PETmean = Periodmean(pet,burnin)
	Packmean = Periodmean(pack,burnin)
	soilmmean = Periodmean(soilm,burnin)
	AETmean = Periodmean(aet,burnin)
	Wmean = {'pet':PETmean,'pack':Packmean,'soilm':soilmmean, 'aet':AETmean}
	return(Wmean)
	
def DataInitialize(BeginYear,EndYear, burnin = 10): #ten year burn in time default
	Pm = Prec(BeginYear,EndYear) #extract climate data for full time period not subtracting burnin period
	Tmax = PRISMextract(BeginYear,EndYear,'tmax')
	Tmin = PRISMextract(BeginYear,EndYear,'tmin')  
	Ta = PRISMextract(BeginYear,EndYear, 'tmean')
	Tdmean = PRISMextract(BeginYear, EndYear, 'tdmean')
	VPsat,VPact,VPD = VapPD(Tdmean, Ta)
	Fm = Meltdata_create(Ta)
	pack = Pack(Fm,Pm)
	Wm,melt,rain = Waterinput(Fm,Pm,pack)
	a,s,e = Topoextract()
	topo = {'aspect':a,'slope':s,'elev':e}
	HL = HeatLI(s,a)
	pet = PET(Ta,HL)
	swhc = SoilWHC()
	soilm, AET = DeltaSoilCalc(swhc,Wm, pet)
	Cgmean, Cbar = Climatesummary(BeginYear,EndYear,burnin) #summary of the climate data will fit the time period considering burn in
	Wmean = Watersummary(pet,pack,soilm,AET,burnin)
	sdata= WBPextract()
	wlis,wf = WLISprep()
	mdata = Datamerge(wlis, wf, sdata)
	out = Datacompile(mdata, Cgmean, Cbar, Wmean)
	return(mdata, Cgmean,Cbar,Wmean,topo,out)

	
	
"""=======================================================================================
==========================================================================================
======================================================================================="""

#code to extract data for FIA 
def writeAOIdata(BeginYear,EndYear,burnin=10, wfile='AOIdata.csv', textwrite = 'n', tifwrite = 'n'):
	Tmax = PRISMextract(BeginYear,EndYear,'tmax')
	Tmin = PRISMextract(BeginYear,EndYear,'tmin')
	Tmean = PRISMextract(BeginYear, EndYear,'tmean')
	Pm = PRISMextract(BeginYear,EndYear,'ppt')
	nrows,ncols = np.shape(Tmax[0].data)
	Tdmean = PRISMextract(BeginYear, EndYear, 'tdmean')
	VPsat,VPact,VPD = VapPD(Tdmean,Tmean)
	Fm = Meltdata_create(Tmean)
	pack = Pack(Fm, Pm)
	Wm, melt,rain = Waterinput(Fm,Pm,pack)
	aspect,slope,ele = Topoextract()
	topo = {'aspect':aspect,'slope':slope,'elev':ele}
	lat,lon = Gridcreate()
	HL = HeatLI(slope,aspect)
	pet = PET(Tmean,HL)
	swhc = SoilWHC()
	soilm, AET = DeltaSoilCalc(swhc,Wm,pet)
	Start = BeginYear+burnin #we want to consider the year after the burnin 
	Tmaxmean, Tmaxbar = PGridmean(Tmax,Start,EndYear) 	#to do, make a function for this to gather all the climate summaries
	Tminmean, Tminbar = PGridmean(Tmin,Start,EndYear) 	
	Pptmean, Pptbar = PGridmean(Pm,Start,EndYear) 		
	Tmeanmean, Tmeanbar= PGridmean(Tmean, Start,EndYear)
	VPDmean, VPDbar = PGridmean(VPD, Start, EndYear)
	AETmean, AETbar = WBGridmean(AET, burnin)
	PETmean, PETbar = WBGridmean(pet, burnin)
	soilmmean, soilmbar = WBGridmean(soilm, burnin)
	packmean, packbar = WBGridmean(pack, burnin)
	deftmean = PETmean-AETmean
	deftbar = PETbar-AETbar
	aetdeftmean = AETmean/deftmean
	aetdeftbar = AETbar/deftbar
	gdd = GDD5(BeginYear+burnin, EndYear)
	
	Cgmean = {'tmax':Tmaxmean,'tmin':Tminmean,'ppt':Pptmean}
	Cbar = {'tmax':Tmaxbar,'tmin':Tminbar,'ppt':Pptbar}
	Wmean = Watersummary(pet,pack,soilm,AET,burnin)
	
	#Write summary of data for FIA here
	filename = 'D:\\Chang\\SE_data\\Climate\\' + wfile
	#gather data into single array
	h =''
	nrows = np.shape(lat)[0]
	ncols = np.shape(lat)[1]
	s = nrows*ncols
	latlon = np.vstack((np.reshape(lat,s), np.reshape(lon,s))).T
	h = h+'lat,lon,'
	
	slopearray = np.reshape(slope, (s,1))
	slopebar = np.mean(slope)
	h = h +'slope,'
	
	aspectarray = np.reshape(aspect, (s,1))
	aspectbar = np.mean(aspect)
	h = h + 'aspect,'
	
	elevarray = np.reshape(ele, (s,1))
	elevbar = np.mean(ele)
	h = h + 'elev,'
	
	swhcarray = np.reshape(swhc,(s,1)) #add soil water holding capacity column
	swhcbar = np.mean(swhc)
	h = h+'swhc,'
	
	HLarray = np.reshape(HL, (s,1))
	HLbar = np.mean(HL)
	h = h+ 'heat_load,'
	
	gddarray = np.zeros((nrows,ncols))
	for i in range(len(gdd)):
		gddarray+= gdd[i]
	gddarray = (gddarray/len(gdd)).reshape((s,1))
	h = h+ 'ggd5,'
	gddbar = np.mean(gddarray)
	
	tmeanarray = []
	for i in range(len(Tmeanmean)):
		tmeanarray.append(Tmeanmean[i].reshape(s))
		h = h+'tmean'+str(i+1)+','
	tmeanarray = np.array(tmeanarray).T
	
	pptarray = []
	for i in range(len(Pptmean)):
		pptarray.append(Pptmean[i].reshape(s))
		h = h+'ppt'+str(i+1)+','
	pptarray = np.array(pptarray).T
	
	tmaxarray = []
	for i in range(len(Tmaxmean)):
		tmaxarray.append(Tmaxmean[i].reshape(s))
		h = h+'tmax'+str(i+1)+','
	tmaxarray = np.array(tmaxarray).T
	
	tminarray = []
	for i in range(len(Tminmean)):
		tminarray.append(Tminmean[i].reshape(s))
		h = h+'tmin'+str(i+1)+','
	tminarray = np.array(tminarray).T
	
	vpdarray = []
	for i in range(len(VPDmean)):
		vpdarray.append(VPDmean[i].reshape(s))
		h = h+'vpd'+str(i+1)+','
	vpdarray = np.array(vpdarray).T
	
	petarray = []
	for i in range(len(PETmean)):
		petarray.append(PETmean[i].reshape(s))
		h = h+'pet'+str(i+1)+','
	petarray = np.array(petarray).T
	
	aetarray = []
	for i in range(len(AETmean)):
		aetarray.append(AETmean[i].reshape(s))
		h = h+'aet'+str(i+1)+','
	aetarray = np.array(aetarray).T
	
	soilmarray = []
	for i in range(len(soilmmean)):
		soilmarray.append(soilmmean[i].reshape(s))
		h = h+'soilm'+str(i+1)+','
	soilmarray = np.array(soilmarray).T
	
	packarray = []
	for i in range(len(packmean)):
		packarray.append(packmean[i].reshape(s))
		h = h+'pack'+str(i+1)+','
	packarray = np.array(packarray).T
	
	deftarray = []
	for i in range(len(deftmean)):
		deftarray.append(deftmean[i].reshape(s))
		h = h+'deft'+str(i+1)+','
	deftarray = np.array(deftarray).T
	
	aetdeftarray = []
	for i in range(len(aetdeftmean)):
		aetdeftarray.append(aetdeftmean[i].reshape(s))
		h = h+'aetdeft'+str(i+1)+','
	aetdeftarray = np.array(aetdeftarray).T
	
	llm = np.array([np.nan, np.nan]) # means for lat and lon
	meansarray = np.hstack((llm, slopebar, aspectbar, elevbar, swhcbar, HLbar, gddbar, Tmeanbar, Pptbar, Tmaxbar, Tminbar, VPDbar, PETbar, AETbar, soilmbar, packbar, deftbar, aetdeftbar))
	
	#dataarray = np.hstack((latlon, slopearray, aspectarray, elevarray, swhcarray, HLarray, gddarray, tmeanarray, pptarray, tmaxarray, tminarray, vpdarray, petarray, aetarray, soilmarray, packarray, deftarray, aetdeftarray))
	dataarray = np.hstack((latlon, slopearray, aspectarray, elevarray, swhcarray, HLarray, gddarray, tmeanarray, pptarray, tmaxarray, tminarray, vpdarray, petarray, aetarray, soilmarray*0.03937, packarray*0.03937, deftarray, aetdeftarray)) #change units from mm to in
	dataarray = np.vstack((meansarray, dataarray)) #add the means line
	if (textwrite == 'y'):
		np.savetxt(filename, dataarray, delimiter =',', header = h)
	if (tifwrite =='y'):
		fiatiffwrite(dataarray,h)

	return(dataarray, h)

def fiatiffwrite(dataarray, h):
	d = np.round(dataarray[1:]) #multiply by 100 and remove the trailing decimals
	d = np.array(d,dtype=int) #change to in datatype
	labels = np.array(h.split(',')[:-1]) #change the header string into an array
	h = Headerextract()
	for i in range(len(labels)):
		Tiffwrite2(np.reshape(d[:,i], (h['nrows'],h['ncols'])), name = labels[i])
	return()
	
	
"""=====================
================================================================================================
====================== JUNKY CODE!!!! FIX THIS AFTER CONFERENCE=====================
==================================================================================="""

def Projecteddataprep(BeginYear,EndYear, rcp, burnin):
	Tmax = GCMextract(BeginYear,EndYear,'tmax', rcp)
	Tmin = GCMextract(BeginYear,EndYear,'tmin',rcp)
	Pm = GCMextract(BeginYear,EndYear,'ppt',rcp)
	Ta = []
	nrows,ncols = np.shape(Tmax[0].data)
	for i in range(len(Tmax)):
		storearray = GCMData(Tmax[i].year, Tmax[i].month, ((Tmax[i].data + Tmin[i].data)/2)) #Definition of Tmean from Daly 2012
		Ta.append(storearray)
	Tdmean = Tmin #==========THIS NEEDS TO BE UPDATED, no Tdmean at the moment so Tmin used 
	VPsat,VPact,VPD = VapPD(Tdmean,Ta)
	Fm = Meltdata_create(Ta)
	pack = Pack(Fm, Pm)
	Wm, melt,rain = Waterinput(Fm,Pm,pack)
	asp,slp,ele = Topoextract()
	aspect = asp[1:,1:]
	slp = slp[1:,1:]
	ele = ele[1:,1:] #GCM data does not cover full domain, off by one row and column
	topo = {'aspect':aspect,'slope':slp,'elev':ele}
	af = (Aspectf(aspect))
	slope = np.radians(slp) #convert to radians
	ymin = 42.262499992719995
	ymax = 46.179166657819998
	lat = np.linspace(ymin,ymax, nrows)
	lat = np.tile(lat,(ncols,1)).T #create latitude grid
	lat = np.radians(lat) #convert to radians
	HL = 0.339 + 0.808*(np.cos(lat)*np.cos(slope)) - 0.196*(np.sin(lat)*np.sin(slope)) - 0.482*(np.cos(af)*np.sin(slope))
	nanIndex = np.isnan(HL) #locate all the nan values (no aspect)
	fillnan = HL[nanIndex] #index of the nan values
	fillnan = 0.339 + 0.808*(np.cos(lat[nanIndex])*np.cos(slope[nanIndex])) - 0.196*(np.sin(lat[nanIndex])*np.sin(slope[nanIndex])) #replace nan values with HL equation without aspect load
	HL[nanIndex] = fillnan #insert values into nan elements
	pet = PET2(Ta,HL)
	swhc = SoilWHC()
	swhc = swhc[1:,1:]
	soilm, AET = DeltaSoilCalc(swhc,Wm,pet)
	Start = BeginYear+burnin #we want to consider the year after the burnin 
	Tmaxmean, Tmaxbar = PGridmean(Tmax,Start,EndYear) 	#to do, make a function for this to gather all the climate summaries
	Tminmean, Tminbar = PGridmean(Tmin,Start,EndYear) 	#
	Pptmean, Pptbar = PGridmean(Pm,Start,EndYear) 		#
	Cgmean = {'tmax':Tmaxmean,'tmin':Tminmean,'ppt':Pptmean}
	Cbar = {'tmax':Tmaxbar,'tmin':Tminbar,'ppt':Pptbar}
	Wmean = Watersummary(pet,pack,soilm,AET,burnin)
	sdata= WBPextract()
	wlis,wf = WLISprep()
	mdata = Datamerge(wlis, wf, sdata)
	return(mdata,Cgmean,Cbar,Wmean, topo)

def PET2(Ta,HL):
    p=[]
    for i in range(len(Ta)):
        p.append(PETcalc2(Ta[i].data,Ta[i].month,HL))
    return(p)

def PETcalc2(Ta,month,hl):    #This is based on the Thornthwaite equation
	ea = 0.611*(np.exp((17.27*Ta)/(Ta+237.3))) #saturation vapour pressure
	dayspmonth = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
	days = np.ones(np.shape(Ta))*dayspmonth[month-1] #create grid of shape(Ta) filled with number of days of given month   
	sd = Solardec()
	ymin = 42.262499992719995
	ymax = 46.179166657819998
	nrows = 471
	ncols =500
	lat = np.linspace(ymin,ymax, nrows)
	lat = np.tile(lat,(ncols,1)).T #create latitude grid
	lat = np.radians(lat) #convert to radians
	#dl = Daylength(sd,lat)
	dl = (Daylength(sd,lat)) #divide by 24 to convert from hours to days <--check this!
	PET = 29.8*days*dl[month-1]*hl*(ea/(Ta+273.2))
	return(PET)

def Gridcreate2():
	xmin = -112.42916667169999
	xmax = -108.27916667336001
	ymin = 42.262499992719995
	ymax = 46.179166657819998
	nrows = 471
	ncols = 500
	x = np.linspace(xmin,xmax,ncols)
	y = np.linspace(ymin,ymax,nrows)
	lon, lat = np.meshgrid(x,y)
	return(lat,lon)

def projdataprep(mdata,Cgmean,Cbar,Wmean,topo):
	lat, lon = Gridcreate2()
	nrows = 471
	ncols = 500
	nn = nrows*ncols
	#reshape arrays to column vectors
	lon =lon.reshape(nn)
	lat =lat.reshape(nn)
	elev = topo['elev'].reshape(nn)
	aet = Wmean['aet'].reshape(nn)
	pet = Wmean['pet'].reshape(nn)
	pack = Wmean['pack'].reshape(nn)
	soilm = Wmean['soilm'].reshape(nn)
	fname = ['lon','lat','elev','AET','PET','pack','soil_m'] #feature names
	testdata = []
	testdata.append(lon)
	testdata.append(lat)
	testdata.append(elev)
	testdata.append(aet)
	testdata.append(pet)
	testdata.append(pack)
	testdata.append(soilm)
	tmonhot = np.argsort(Cbar['tmax'])[:8:-1] #3 hottest months
	tmoncold = np.argsort(Cbar['tmin'])[:3] #3 coldest months
	pptwet = np.argsort(Cbar['ppt'])[:8:-1] #3 wettest months
	pptdry = np.argsort(Cbar['ppt'])[:3] #3 driest months
	for i in range(3):													#Ugly coding, but for formmating sake....
		testdata.append(Cgmean['ppt'][pptdry[i]].reshape(nn))
		fname.append('Pdry'+str(i+1)+'_mon'+str(pptdry[i]+1))
	for i in range(3):
		testdata.append(Cgmean['ppt'][pptwet[i]].reshape(nn))
		fname.append('Pwet'+str(i+1)+'_mon'+str(pptwet[i]+1))
	for i in range(3):
		testdata.append(Cgmean['tmax'][tmonhot[i]].reshape(nn))
		fname.append('Tmax'+str(i+1)+'_mon'+str(tmonhot[i]+1))
	for i in range(3):
		testdata.append(Cgmean['tmin'][tmoncold[i]].reshape(nn))
		fname.append('Tmin'+str(i+1)+'_mon'+str(tmoncold[i]+1))
	testdata = np.array(testdata)
	fname = np.array(fname)
	return(testdata.T,fname)	

def writeprojdata(testdata,fname,filename):
	head = ','.join(fname)
	np.savetxt('D:\\chang\\python_scripts\\output\\projections2\\'+filename+'.csv',testdata,delimiter=',',header=head)
	return()
	
"""==================================================================================
======================MAIN===========================================================
=================================================================================="""
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

#generate the test and sample dataset
sampledata = np.genfromtxt('D:\\chang\\python_scripts\\output\\samples\\sample04102013.csv', delimiter = ',', names=True)
testdata = np.genfromtxt('D:\\chang\\python_scripts\\output\\testdata_03282013.csv', delimiter =',', names=True)

def RFadult(sampledata,testdata):
#sort out where the seedlings are not nan
	snanind = ~np.isnan(sampledata['seedl'])
	y_rp = sampledata['reprod'][snanind]
	y_rp = np.where(y_rp >0, 1, 0)
	x_rp = []
	for i in range(8,len(sampledata.dtype.names)):
		x_rp.append(sampledata[sampledata.dtype.names[i]][snanind])
	x_rp = np.array(x_rp).T #change to np array and transpose
	#generate test data
	xt_rp = []
	for i in range(3,len(testdata.dtype.names)):
		xt_rp.append(testdata[testdata.dtype.names[i]])
	xt_rp = np.array(xt_rp).T

	h = Headerextract()
	nrow= h['nrows']
	ncol= h['ncols']
	#clasf = RandomForestClassifier(n_estimators=1000, criterion = 'gini',compute_importances = True, max_features=20,min_samples_leaf=30, oob_score=True, bootstrap = True)
	clasf = RandomForestClassifier(n_estimators=2000, criterion = 'gini',compute_importances = True, max_features=8,min_samples_leaf=20, oob_score=True, bootstrap = True)
	rfmodel = clasf.fit(x_rp, y_rp) #fit model with random forest classifier for presence case
	ytestprob = rfmodel.predict_proba(xt_rp)
	ytestprob_0 = ytestprob[:,0].reshape((nrow,ncol))
	ytestprob_1 = ytestprob[:,1].reshape((nrow,ncol))

	#feature importance
	testxfname= np.array(testdata.dtype.names[3:])
	featureimp = rfmodel.feature_importances_
	sxfeature = testxfname[np.argsort(featureimp)][::-1]
	sfeatureimp = np.sort(featureimp)[::-1]

	#confusion matrix
	ypred = rfmodel.predict(x_rp)
	cmx = confusion_matrix(y_rp, ypred)
	cmxperc = np.array([cmx[0]/np.sum(cmx[0]), cmx[1]/np.sum(cmx[1])])

	#roc curve
	ypred_prob = rfmodel.predict_proba(x_rp)
	fpr, tpr, thresholds = roc_curve(y_rp, ypred_prob[:,1])
	roc_auc = auc(fpr,tpr)

	#====plot routines====
	#feature importance plot
	plt.subplot2grid((3,3),(0,2), rowspan=2)
	pos = np.arange(len(sxfeature))+0.5
	plt.barh(pos, sfeatureimp[::-1], align='center')
	plt.yticks(pos, sxfeature[::-1])
	plt.xlabel('Mean decrease in accuracy')
	plt.ylabel('Feature')
	plt.title('Feature importance plot')
	plt.grid()
	#plt.show()

	#roc curve plot
	plt.subplot2grid((3,3),(2,2))
	plt.plot(fpr,tpr,label = 'ROC curve (area = %0.2f)' %roc_auc)
	plt.plot([0,1],[0,1], 'k--')
	plt.xlim([-0.01,1.01])
	plt.ylim([-0.01,1.01])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic plot')
	plt.grid()
	plt.legend(loc='lower right')
	#plt.show()

	#probability map plot
	xmin = -112.438
	xmax = -108.271
	ymin = 42.262
	ymax = 46.187
	ae = [xmin, xmax,ymin, ymax]
	lat = sampledata['lat'][snanind]
	lon = sampledata['long'][snanind]
	lat1 = lat[np.where(y_rp==1)]
	lon1 = lon[np.where(y_rp==1)]
	lat0 = lat[np.where(y_rp==0)]
	lon0 = lon[np.where(y_rp==0)]
	
	plt.subplot2grid((3,3),(0,0),colspan =2, rowspan=3)
	pp = plt.imshow(ytestprob_1, extent = ae)
	cbar = plt.colorbar(pp)
	cbar.set_label('Probability of presence')
	p1 = plt.scatter(lon1,lat1, s=20, marker ='x', color = 'green', alpha = 0.9, label = 'field presence')
	p0 = plt.scatter(lon0,lat0, s= 7, marker = 'o', color='gray', alpha = 0.3, label = 'field absence')
	plt.xlabel('Longitude (DD)')
	plt.ylabel('Latitude (DD)')
	plt.legend((p1,p0), (p1.get_label(), p0.get_label()),loc ='upper right')
	plt.title("RF modeled Whitebark pine species distribution map for >=8\" DBH individuals")
	plt.grid()
	plt.subplots_adjust(wspace=0.4, hspace=0.5)
	plt.show()
	
	return(cmx,cmxperc)

def RFadult_limited(sampledata,testdata):
#sort out where the seedlings are not nan
	snanind = ~np.isnan(sampledata['seedl'])
	y_rp = sampledata['reprod'][snanind]
	y_rp = np.where(y_rp >0, 1, 0)
	x_rp = []
	
	for i in range(8,24):
		x_rp.append(sampledata[sampledata.dtype.names[i]][snanind])
	x_rp = np.array(x_rp).T #change to np array and transpose
	#generate test data
	xt_rp = []
	for i in range(3,19):
		xt_rp.append(testdata[testdata.dtype.names[i]])
	xt_rp = np.array(xt_rp).T

	h = Headerextract()
	nrow= h['nrows']
	ncol= h['ncols']
	clasf = RandomForestClassifier(n_estimators=2000, criterion = 'gini',compute_importances = True, max_features=8,min_samples_leaf=20, oob_score=True, bootstrap = True)
	#clasf = RandomForestClassifier(n_estimators=2000, criterion = 'gini',compute_importances = True, max_features=4,min_samples_leaf=20, oob_score=True, bootstrap = True)
	rfmodel = clasf.fit(x_rp, y_rp) #fit model with random forest classifier for presence case
	ytestprob = rfmodel.predict_proba(xt_rp)
	ytestprob_0 = ytestprob[:,0].reshape((nrow,ncol))
	ytestprob_1 = ytestprob[:,1].reshape((nrow,ncol))

	#feature importance
	testxfname= np.array(testdata.dtype.names[3:])
	featureimp = rfmodel.feature_importances_
	sxfeature = testxfname[np.argsort(featureimp)][::-1]
	sfeatureimp = np.sort(featureimp)[::-1]

	#confusion matrix
	ypred = rfmodel.predict(x_rp)
	cmx = confusion_matrix(y_rp, ypred)
	cmxperc = np.array([cmx[0]/np.sum(cmx[0]), cmx[1]/np.sum(cmx[1])])

	#roc curve
	ypred_prob = rfmodel.predict_proba(x_rp)
	fpr, tpr, thresholds = roc_curve(y_rp, ypred_prob[:,1])
	roc_auc = auc(fpr,tpr)

	#====plot routines====
	#feature importance plot
	plt.subplot2grid((3,3),(0,2), rowspan=2)
	pos = np.arange(len(sxfeature))+0.5
	plt.barh(pos, sfeatureimp[::-1], align='center')
	plt.yticks(pos, sxfeature[::-1])
	plt.xlabel('Mean decrease in accuracy')
	plt.ylabel('Feature')
	plt.title('Feature importance plot')
	plt.grid()
	#plt.show()

	#roc curve plot
	plt.subplot2grid((3,3),(2,2))
	plt.plot(fpr,tpr,label = 'ROC curve (area = %0.2f)' %roc_auc)
	plt.plot([0,1],[0,1], 'k--')
	plt.xlim([-0.01,1.01])
	plt.ylim([-0.01,1.01])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic plot')
	plt.grid()
	plt.legend(loc='lower right')
	#plt.show()

	#probability map plot
	xmin = -112.438
	xmax = -108.271
	ymin = 42.262
	ymax = 46.187
	ae = [xmin, xmax,ymin, ymax]
	lat = sampledata['lat'][snanind]
	lon = sampledata['long'][snanind]
	lat1 = lat[np.where(y_rp==1)]
	lon1 = lon[np.where(y_rp==1)]
	lat0 = lat[np.where(y_rp==0)]
	lon0 = lon[np.where(y_rp==0)]
	
	plt.subplot2grid((3,3),(0,0),colspan =2, rowspan=3)
	pp = plt.imshow(ytestprob_1, extent = ae)
	cbar = plt.colorbar(pp)
	cbar.set_label('Probability of presence')
	p1 = plt.scatter(lon1,lat1, s=20, marker ='x', color = 'green', alpha = 0.9, label = 'field presence')
	p0 = plt.scatter(lon0,lat0, s= 7, marker = 'o', color='gray', alpha = 0.3, label = 'field absence')
	plt.xlabel('Longitude (DD)')
	plt.ylabel('Latitude (DD)')
	plt.legend((p1,p0), (p1.get_label(), p0.get_label()),loc ='upper right')
	plt.title("RF modeled Whitebark pine species distribution map for >=8\" DBH individuals")
	plt.grid()
	plt.subplots_adjust(wspace=0.4, hspace=0.5)
	plt.show()
	
	return(cmx,cmxperc)

def RFmort(sampledata,testdata):
#sort out where the seedlings are not nan
	snanind = ~np.isnan(sampledata['mort'])
	y_rp = sampledata['mort'][snanind]
	y_rp = np.where(y_rp >0, 1, 0)
	x_rp = []
	for i in range(8,len(sampledata.dtype.names)):
		x_rp.append(sampledata[sampledata.dtype.names[i]][snanind])
	x_rp = np.array(x_rp).T #change to np array and transpose
	#generate test data
	xt_rp = []
	for i in range(3,len(testdata.dtype.names)):
		xt_rp.append(testdata[testdata.dtype.names[i]])
	xt_rp = np.array(xt_rp).T

	h = Headerextract()
	nrow= h['nrows']
	ncol= h['ncols']
	clasf = RandomForestClassifier(n_estimators=1000, criterion = 'gini',compute_importances = True, 
	max_features=20,min_samples_leaf=30, oob_score=True, bootstrap = True)
	rfmodel = clasf.fit(x_rp, y_rp) #fit model with random forest classifier for presence case
	ytestprob = rfmodel.predict_proba(xt_rp)
	ytestprob_0 = ytestprob[:,0].reshape((nrow,ncol))
	ytestprob_1 = ytestprob[:,1].reshape((nrow,ncol))

	#feature importance
	testxfname= np.array(testdata.dtype.names[3:])
	featureimp = rfmodel.feature_importances_
	sxfeature = testxfname[np.argsort(featureimp)][::-1]
	sfeatureimp = np.sort(featureimp)[::-1]

	#confusion matrix
	ypred = rfmodel.predict(x_rp)
	cmx = confusion_matrix(y_rp, ypred)
	cmxperc = np.array([cmx[0]/np.sum(cmx[0]), cmx[1]/np.sum(cmx[1])])

	#roc curve
	ypred_prob = rfmodel.predict_proba(x_rp)
	fpr, tpr, thresholds = roc_curve(y_rp, ypred_prob[:,1])
	roc_auc = auc(fpr,tpr)
	
	#====plot routines====
	#feature importance plot
	plt.subplot2grid((3,3),(0,2), rowspan=2)
	pos = np.arange(len(sxfeature))+0.5
	plt.barh(pos, sfeatureimp[::-1], align='center')
	plt.yticks(pos, sxfeature[::-1])
	plt.xlabel('Mean decrease in accuracy')
	plt.ylabel('Feature')
	plt.title('Feature importance plot')
	plt.grid()
	#plt.show()

	#roc curve plot
	plt.subplot2grid((3,3),(2,2))
	plt.plot(fpr,tpr,label = 'ROC curve (area = %0.2f)' %roc_auc)
	plt.plot([0,1],[0,1], 'k--')
	plt.xlim([-0.01,1.01])
	plt.ylim([-0.01,1.01])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic plot')
	plt.grid()
	plt.legend(loc='lower right')
	#plt.show()

	#probability map plot
	xmin = -112.438
	xmax = -108.271
	ymin = 42.262
	ymax = 46.187
	ae = [xmin, xmax,ymin, ymax]
	lat = sampledata['lat'][snanind]
	lon = sampledata['long'][snanind]
	lat1 = lat[np.where(y_rp==1)]
	lon1 = lon[np.where(y_rp==1)]
	lat0 = lat[np.where(y_rp==0)]
	lon0 = lon[np.where(y_rp==0)]
	
	plt.subplot2grid((3,3),(0,0),colspan =2, rowspan=3)
	pp = plt.imshow(ytestprob_1, extent = ae)
	cbar = plt.colorbar(pp)
	cbar.set_label('Probability of presence')
	p1 = plt.scatter(lon1,lat1, s=20, marker ='x', color = 'green', alpha = 0.9, label = 'field presence')
	p0 = plt.scatter(lon0,lat0, s= 7, marker = 'o', color='gray', alpha = 0.3, label = 'field absence')
	plt.xlabel('Longitude (DD)')
	plt.ylabel('Latitude (DD)')
	plt.legend((p1,p0), (p1.get_label(), p0.get_label()),loc ='upper right')
	plt.title("RF modeled Whitebark pine species distribution map for >=8\" DBH individuals (Mortality)")
	plt.grid()
	plt.subplots_adjust(wspace=0.4, hspace=0.5)
	plt.show()
	
	return(cmx,cmxperc)

"""==============================================================================================
==================================RF for projections=============================================
=============================================================================================="""
def RFprojections(sampledata):
#sort out where the seedlings are not nan
	snanind = ~np.isnan(sampledata['seedl'])
	y_rp = sampledata['reprod'][snanind]
	y_rp = np.where(y_rp >0, 1, 0)
	x_rp = []
	
	for i in range(8,24):
		x_rp.append(sampledata[sampledata.dtype.names[i]][snanind])
	x_rp = np.array(x_rp).T #change to np array and transpose
	#generate test data
	testdata = np.genfromtxt('D:\\chang\\python_scripts\\output\\testdata_03282013.csv', delimiter =',', names=True) #current time period	
	tdata1 = np.genfromtxt('d:\\chang\\python_scripts\\output\\projections2\\testdata_rcp45_cesm1_bgc_2010_2040.csv',delimiter=',',names=True)
	tdata2 = np.genfromtxt('d:\\chang\\python_scripts\\output\\projections2\\testdata_rcp45_cesm1_bgc_2040_2070.csv',delimiter=',',names=True)
	tdata3 = np.genfromtxt('d:\\chang\\python_scripts\\output\\projections2\\testdata_rcp45_cesm1_bgc_2070_2100.csv',delimiter=',',names=True)
	xt_rp = []
	xt_rp1 = []
	xt_rp2 = []
	xt_rp3 = []
	
	for i in range(3,19):
		xt_rp.append(testdata[testdata.dtype.names[i]])
	xt_rp = np.array(xt_rp).T
	
	for i in range(3,len(tdata1.dtype.names)):
		xt_rp1.append(tdata1[tdata1.dtype.names[i]])
		xt_rp2.append(tdata2[tdata2.dtype.names[i]])
		xt_rp3.append(tdata3[tdata3.dtype.names[i]])
	xt_rp1 = np.array(xt_rp1).T
	xt_rp2 = np.array(xt_rp2).T
	xt_rp3 = np.array(xt_rp3).T

	h = Headerextract()
	nrow= h['nrows']
	ncol= h['ncols']
	#clasf = RandomForestClassifier(n_estimators=2000, criterion = 'gini',compute_importances = True, max_features=8,min_samples_leaf=30, oob_score=True, bootstrap = True)
	clasf = RandomForestClassifier(n_estimators=2000, criterion = 'gini',compute_importances = True, max_features=8,min_samples_leaf=20, oob_score=True, bootstrap = True)
	rfmodel = clasf.fit(x_rp, y_rp) #fit model with random forest classifier for presence case
	ytestprob = rfmodel.predict_proba(xt_rp)
	ytestprob_0 = ytestprob[:,0].reshape((nrow,ncol))
	ytestprob_1 = ytestprob[:,1].reshape((nrow,ncol))

	#====plot routines====
	
	#probability map plot of current distribution
	xmin = -112.438
	xmax = -108.271
	ymin = 42.262
	ymax = 46.187
	ae = [xmin, xmax,ymin, ymax]
	lat = sampledata['lat'][snanind]
	lon = sampledata['long'][snanind]
	lat1 = lat[np.where(y_rp==1)]
	lon1 = lon[np.where(y_rp==1)]
	lat0 = lat[np.where(y_rp==0)]
	lon0 = lon[np.where(y_rp==0)]
	
	plt.subplot2grid((2,2),(0,0))
	pp = plt.imshow(ytestprob_1, extent = ae, vmin = 0, vmax = .85)
	cbar = plt.colorbar(pp)
	cbar.set_label('Probability of presence')
	#p1 = plt.scatter(lon1,lat1, s=20, marker ='x', color = 'green', alpha = 0.9, label = 'field presence')
	#p0 = plt.scatter(lon0,lat0, s= 7, marker = 'o', color='gray', alpha = 0.3, label = 'field absence')
	plt.xlabel('Longitude (DD)')
	plt.ylabel('Latitude (DD)')
	plt.xticks(fontsize=9)
	plt.yticks(fontsize=9)
	#plt.legend((p1,p0), (p1.get_label(), p0.get_label()),loc ='upper right')
	plt.title("RF modeled WBP distribution for >=8\" DBH individuals year 2012")
	plt.grid()
	
	
	#probability map plots for future distributions
	x_proj = [xt_rp1,xt_rp2,xt_rp3]
	y_proj1 = []
	y_proj0 = []
	pnrow = 471
	pncol = 500
	for i in range(len(x_proj)):
		ytestproj = rfmodel.predict_proba(x_proj[i])
		y_proj0.append(ytestproj[:,0].reshape((pnrow,pncol)))
		y_proj1.append(ytestproj[:,1].reshape((pnrow,pncol)))
	
	pxmin = -112.42916667169999
	pxmax = -108.27916667336001
	pymin = 42.262499992719995
	pymax = 46.179166657819998
	pae = [pxmin,pxmax,pymin,pymax]
	
	gridtuple = [(0,1),(1,0),(1,1)]
	subtitles = ['CESM-1 BGC RCP-8.5 Projection year 2040','CESM-1 BGC RCP-8.5 Projection year 2070','CESM-1 BGC RCP-8.5 Projection 2100']
	for i in range(len(y_proj1)):
		plt.subplot2grid((2,2),gridtuple[i])
		prim = plt.imshow(y_proj1[i], extent =pae, vmin = 0, vmax = .85)
		cbar = plt.colorbar(prim)
		cbar.set_label('Probability of presence')
		plt.xlabel('Longitude (DD)')
		plt.ylabel('Latitude (DD)')
		plt.xticks(fontsize=9)
		plt.yticks(fontsize=9)
		plt.title(subtitles[i])
		plt.grid(True,alpha=0.5)
	
	plt.subplots_adjust(hspace = 0.2,wspace=0.01)
	plt.show()
	yp = [np.reshape(ytestprob_1, 472*501)]
	for i in range(3):
		yp.append(np.reshape(y_proj1[i], pnrow*pncol))
	return(yp)

def RFprobhisto(yp, th = 0.1):
	years = [2040,2070,2100]
	probsums = []
	for i in range(len(yp)):
		probsums.append(len(np.where(yp[i]>0.1)[0]))
		plt.subplot(2,2,i+1)
		plt.hist(yp[i][np.where(yp[i]>0.1)])
		if i ==0:
			plt.title('Present probability histogram')
		else:
			plt.title('Projection ' + str(years[i-1]) +' probability histogram')
		plt.xlabel('Probability bin')
		plt.ylabel('Pixel frequency')
		plt.grid(True, alpha = 0.5)
	plt.show()
	probsums = np.array(probsums)
	return(probsums)
	
#=============================================================================
#=============================================================================
def RFmortprojection(sampledata):
#sort out where the seedlings are not nan
	snanind = ~np.isnan(sampledata['mort'])
	y_rp = sampledata['mort'][snanind]
	y_rp = np.where(y_rp >0, 1, 0)
	x_rp = []
	for i in range(8,24):
		x_rp.append(sampledata[sampledata.dtype.names[i]][snanind])
	x_rp = np.array(x_rp).T #change to np array and transpose
	#generate test data
	
	testdata = np.genfromtxt('D:\\chang\\python_scripts\\output\\testdata_03282013.csv', delimiter =',', names=True) #current time period	
	tdata1 = np.genfromtxt('d:\\chang\\python_scripts\\output\\projections2\\testdata_rcp85_cesm1_bgc_2010_2040.csv',delimiter=',',names=True)
	tdata2 = np.genfromtxt('d:\\chang\\python_scripts\\output\\projections2\\testdata_rcp85_cesm1_bgc_2040_2070.csv',delimiter=',',names=True)
	tdata3 = np.genfromtxt('d:\\chang\\python_scripts\\output\\projections2\\testdata_rcp85_cesm1_bgc_2070_2100.csv',delimiter=',',names=True)
	xt_rp = []
	xt_rp1 = []
	xt_rp2 = []
	xt_rp3 = []
	
	for i in range(3,19):
		xt_rp.append(testdata[testdata.dtype.names[i]])
	xt_rp = np.array(xt_rp).T
	
	for i in range(3,len(tdata1.dtype.names)):
		xt_rp1.append(tdata1[tdata1.dtype.names[i]])
		xt_rp2.append(tdata2[tdata2.dtype.names[i]])
		xt_rp3.append(tdata3[tdata3.dtype.names[i]])
	xt_rp1 = np.array(xt_rp1).T
	xt_rp2 = np.array(xt_rp2).T
	xt_rp3 = np.array(xt_rp3).T

	h = Headerextract()
	nrow= h['nrows']
	ncol= h['ncols']
	clasf = RandomForestClassifier(n_estimators=1000, criterion = 'gini',compute_importances = True, 
	max_features=2,min_samples_leaf=20, oob_score=True, bootstrap = True)
	rfmodel = clasf.fit(x_rp, y_rp) #fit model with random forest classifier for presence case
	ytestprob = rfmodel.predict_proba(xt_rp)
	ytestprob_0 = ytestprob[:,0].reshape((nrow,ncol))
	ytestprob_1 = ytestprob[:,1].reshape((nrow,ncol))

	#probability map plot
	x_proj = [xt_rp1,xt_rp2,xt_rp3]
	y_proj1 = []
	y_proj0 = []
	pnrow = 471
	pncol = 500
	for i in range(len(x_proj)):
		ytestproj = rfmodel.predict_proba(x_proj[i])
		y_proj0.append(ytestproj[:,0].reshape((pnrow,pncol)))
		y_proj1.append(ytestproj[:,1].reshape((pnrow,pncol)))
	
	pxmin = -112.42916667169999
	pxmax = -108.27916667336001
	pymin = 42.262499992719995
	pymax = 46.179166657819998
	pae = [pxmin,pxmax,pymin,pymax]
	
	gridtuple = [(0,1),(1,0),(1,1)]
	subtitles = ['CESM-1 BGC RCP-8.5 Mortality Projection year 2040','CESM-1 BGC RCP-8.5 Projection year 2070','CESM-1 BGC RCP-8.5 Projection 2100']
	for i in range(len(y_proj1)):
		plt.subplot2grid((2,2),gridtuple[i])
		prim = plt.imshow(y_proj1[i], extent =pae, vmin = 0, vmax = .85)
		cbar = plt.colorbar(prim)
		cbar.set_label('Probability of presence')
		plt.xlabel('Longitude (DD)')
		plt.ylabel('Latitude (DD)')
		plt.xticks(fontsize=9)
		plt.yticks(fontsize=9)
		plt.title(subtitles[i])
		plt.grid(True,alpha=0.5)
	
	plt.subplots_adjust(hspace = 0.2,wspace=0.01)
	plt.show()
	
	return()
'''===================OLD MAIN CODE========================
import time
start_time = time.clock()
BeginYear1 = 1940 #make sure that this value is 10 less, as we need to consider a decade of burn in time for the model
EndYear1 = 1980
mdata,Cgmean,Cbar,Wmean,topo,out = DataInitialize(BeginYear1,EndYear1)
BeginYear2 = 1970 #make sure that this value is 10 less, as we need to consider a decade of burn in time for the model
EndYear2 = 2010
mdata2,Cgmean2,Cbar2,Wmean2,topo2,out2 = DataInitialize(BeginYear2,EndYear2)

# generaate sample data
sampledata, fname = Sampleprep(out,out2) #works for 2 time period data set	
#generate test data
testdata, testfname = Testprep(Cgmean,Cbar,Wmean,topo)
testdata2, testfname2 = Testprep(Cgmean2,Cbar2,Wmean2,topo2)
y,ylabel,x,xfname, testset,testfeat= RFdataprep(sampledata,fname,testdata,testfname,testdata2,testfname2)
xtest = testset[:,3:] #did not include elevation for this test
testxfname = testfeat[3:]
testxfname = np.array(testxfname)

end_time = time.clock()
print(str((end_time-start_time)) + ' seconds to process')


y[np.where(y==1)] = 0
y[np.where(y==2)] = 1
import sklearn
from sklearn.ensemble import RandomForestClassifier
nrow= 472
ncol=501
clf = RandomForestClassifier(n_estimators=1000, criterion = 'gini',compute_importances = True, 
max_features=20,min_samples_leaf=30, oob_score=True, bootstrap = True)
rfmodel = clf.fit(x, y) #fit model with random forest classifier for presence case
ytest = rfmodel.predict(xtest)
yplt = ytest.reshape((nrow,ncol))
ytestprob = rfmodel.predict_proba(xtest)
ytestprob_0 = ytestprob[:,0].reshape((nrow,ncol))
ytestprob_1 = ytestprob[:,1].reshape((nrow,ncol))
#ytestprob_2 = ytestprob[:,2].reshape((nrow,ncol))
featureimp = rfmodel.feature_importances_
sxfeature = testxfname[np.argsort(featureimp)][::-1]
sfeatureimp = np.sort(featureimp)[::-1]

from sklearn.metrics import confusion_matrix
ypred = rfmodel.predict(x)
cmx = confusion_matrix(y, ypred)
cmxperc = np.array([cm[0]/np.sum(cm[0]), cm[1]/np.sum(cm[1])])

sampleframe = pd.DataFrame(x,index = cn, columns =labels)
scores = cross_val_scores(clf,x,y,cv=10) #10 fold cross validation
'''

"""=======================================================================
===========================visualization funcitons========================
======================================================================="""
def varimportplot(sf, sfimport):
	pos = np.arange(len(sf))+.5
	pos = pos[::-1]
	plt.barh(pos, sf, align='center')
	plt.yticks(pos, sf)
	plt.xlabel('Feature importance')
	plt.grid()
	plt.title('Feature importance plot')
	plt.show()
	return()
	
def Ani(data):
	fig = plt.figure(0)
	ims = []
	for i in range(len(data)):
		im = plt.imshow(data[i])
		ims.append([im]) #save each plot
	ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,repeat_delay=1000) #send plot to animator
	#ani.save('D:\\chang\\python_scripts\\output\\output_movie.mp4')
	plt.show(0)
	

def IPlot(p):
    fig = plt.figure()
    #fig, axes = plt.subplots(3,4)
    #im = plt.imshow(p, cmap = 'cool') 
    for i in range(len(p)):
        plt.subplot(3,4,i+1)
        #axes[i].subplot(...
        norm = mpl.colors.Normalize(vmin = 0, vmax = np.mean(p)+np.std(p))
        im = plt.imshow(p[i], cmap = 'cool') 
        im.set_norm(norm)
        plt.colorbar(im)
        #plt.show()
        plt.title('month ' + str(i))
    fig.show()
    return()
