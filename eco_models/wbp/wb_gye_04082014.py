# -*- coding: utf-8 -*-
"""
#Created on Tue Aug 20 12:32:55 2013

#@author: tony.chang
#Climatic water deficit equations from 
#Lutz, J.A., van Wagtendonk, J.W., and Franklin, J.F. Climatic water deficit, 
#tree species, and climate change in Yosemite National Park. 2010. 
#Journal of Biogeography
#
#Dependencies: Gridded data products are extracted in TIFF and NetCDF4 format.
#
#re-run for the GYE with new boundary box 01092014
#TO DO: work on the GCM portion to calculate water balance for all GCM and scenarios

"""
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
	def __init__(self, year=None, month=None, data=None):
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

def PRISMextract(BeginYear,EndYear,var):        
    Pdata = [] #List to store all PRISMData object
    workspace = "E:\\PRISM\\" + var +"\\"    
    for searchyear in range(BeginYear, EndYear+1): #looping through years of interest
        for filenum in range(1,13): #does not consider the annual mean filenum (#14)
            filename = workspace + "PRISM800m_" + var + str(searchyear) + "_" + str(filenum) + ".tif"
            readfile =  gdal.Open(filename)
            data = np.array(readfile.GetRasterBand(1).ReadAsArray())
            x = GridData(searchyear,filenum,data) #Create instance of PRISMData object
            Pdata.append(x) 
            readfile = None #close file
    return(Pdata)

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
	workspace = "E:\\CMIP5\\GCM\\" + model + "\\"
	GCMdata = []
	for cyear in range(beginyear,endyear+1):
		for month in range(1,13):
			if rcp =='historical':
				filename = workspace + 'historical' + '\\' + v + '\\' + model + '_historical' + '_' + v + '_' +str(cyear) + '_' +str(month) +'.tif'
			else:
				filename = workspace + 'rcp' + str(rcp) + '\\' + v + '\\' + model + '_rcp' + str(rcp) + '_' + v + '_' +str(cyear) + '_' +str(month) +'.tif'
			readfile = gdal.Open(filename)
			data = np.array(readfile.GetRasterBand(1).ReadAsArray())
			if (v=='pr'):
				multiplier = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
				data = multiplier[month-1] * data * 86400 # convert from kg m^-2 s^-1 to mm/month (kg/m2 ~= mm)
			else:
				data = data - 273.15 #convert from K to C
			x = GridData(cyear,month-1,data)
			GCMdata.append(x)
			readfile = None #close file
	return(GCMdata)

def WBextract(BeginYear,EndYear,var):        
	Pdata = [] #List to store all PRISMData object
	workspace = "E:\\WB\\"+ var +"\\"    
	for searchyear in range(BeginYear, EndYear+1): #looping through years of interest
		for filenum in range(1,13): #does not consider the annual mean filenum (#14)
			filename = workspace + var + "_" + str(searchyear) + "_" + str(filenum) + ".tif"
			readfile =  gdal.Open(filename)
			data = np.array(readfile.GetRasterBand(1).ReadAsArray())
			x = GridData(searchyear,filenum,data) #Create instance of PRISMData object
			Pdata.append(x) 
			readfile = None #close file
	return(Pdata)
	
def MeanTemp(BeginYear,EndYear, rcp, mod):
#Average monthly temperature function for GCMs
    Tmax = GCMextract(BeginYear,EndYear,'tmax', rcp, mod)
    Tmin = GCMextract(BeginYear,EndYear,'tmin', rcp, mod)    
    Tmean = []
    for i in range(len(Tmax)):
        storearray = GridData(Tmax[i].year, Tmax[i].month, ((Tmax[i].data + Tmin[i].data)/2)) #Definition of Tmean from Daly 2012
        Tmean.append(storearray)
    return(Tmean)

def Topoextract():
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

def tiffextract(path):
	ds = gdal.Open(path)
	return(np.array(ds.GetRasterBand(1).ReadAsArray()))

#maskpath  = "D:\\CHANG\\PhD_Material\\Climate_Project\\gyepace_mask.tif"
def maskedaverage(data,maskpath):
	m = np.where(tiffextract(maskpath)==1,0,1) #extract mask and change zeros to ones and ones to zeros
	muy = np.zeros(len(data))
	for i in range(len(data)):
		muy[i] = (np.ma.masked_array(data[i].data, mask=m)).mean()
	return(muy)
	
#===============================================================================
#========================Utilities==============================================
#===============================================================================
 
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

def Gridcreate(gcm='n'):
    header = Headerextract(gcm=gcm)
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

#===============================================================================
#====================Water Balance Models=======================================    
#===============================================================================

def VapPD(Tdmean, Tmean): 
#vapor pressure calculations from Campbell and Norman 1998, adapted from Weiss et al 2012
    a = 0.611 #kPa
    b = 17.502
    c = 240.97 # deg C
    VPsat = []
    VPact = []
    VPDmean = []
    for i in range(len(Tmean)):    
        VPsatstorearray = GridData(Tmean[i].year, Tmean[i].month, a*np.exp(((b*Tmean[i].data)/(Tmean[i].data+c))))
        VPactstorearray = GridData(Tmean[i].year, Tmean[i].month, a*np.exp(((b*(Tdmean[i].data))/((Tdmean[i].data)+c))))
        VPDmeanstorearray = GridData(Tmean[i].year, Tmean[i].month, VPsatstorearray.data-VPactstorearray.data)         
        VPsat.append(VPsatstorearray)
        VPact.append(VPactstorearray)
        VPDmean.append(VPDmeanstorearray)
    return(VPsat,VPact,VPDmean)
	
def Meltfactor(Ta):  
    Fm = Ta
    Fm[Fm<0] = 0 #first test if Ta <= 0
    Fm = Fm*0.167    
    Fm[Fm>1.002] = 1 #second test if Ta >= 6*0.167, set equal to 1 
    return(Fm)
    
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
    dl = (Daylength(sd,lat)) #divide by 24 to convert from hours to days <--check this!
    PET = 29.8*days*dl[month-1]*hl*(ea/(Ta+273.2))
    return(PET)

def PET(Ta,HL):
    p=[]
    for i in range(len(Ta)):
        p.append(PETcalc(Ta[i].data,Ta[i].month,HL))
    return(p)

def DeltaSoilCalc(soilmax,Wm,pet):#calculate the soil moisture, additionally add a runoff calculations
	soilm = []
	AETarray = []
	runoffarray = []
	for i in range(len(pet)):   
		if (i ==0):
			soilm.append(np.ones(np.shape(pet[i]))*1e-4) #start soil moisture at zero
			AETarray.append(pet[i])
			runoffarray.append(np.zeros(np.shape(pet[i])))
		else:
			#calculate 2 seperate cases for AET, based on the relationship of Wm and PET
			#case 1 if Wm>PET
			tempAET1 = pet[i]
			tempsoilmoisture1 = np.minimum(Wm[i]-pet[i]+soilm[i-1], soilmax)
			tempsoilrunoff1 = np.maximum(0,Wm[i]-pet[i]+soilm[i-1]-soilmax)
			#case 2 if Wm<=PET
			dsoil = soilm[i-1] * (1- np.exp((Wm[i]- pet[i])/soilmax))
			tempAET2 = Wm[i]+ dsoil
			tempsoilmoisture2 = np.minimum(Wm[i] - tempAET2 +soilm[i-1], soilmax) #water for the month is the water that came in, what came out, plus what was before          
			tempsoilrunoff2 = np.maximum(0,Wm[i] - tempAET2 +soilm[i-1] - soilmax) #amount of water that is runoff
			#now run the conditions on the arrays
			AET = np.where(Wm[i]>pet[i], tempAET1, tempAET2)
			soilmoisture = np.where(Wm[i]>pet[i], tempsoilmoisture1, tempsoilmoisture2) #where moisture is greater than PET
			runoff = np.where(Wm[i]>pet[i], tempsoilrunoff1, tempsoilrunoff2)
			runoffarray.append(runoff)
			soilm.append(soilmoisture) #the culmulative soil moisture for the time step
			AETarray.append(AET)        
	return(soilm,AETarray, runoffarray)
	
def SoilWHC(): #defines the soil water holding capacity, currently using STATSGO dataset with depth at 150cm
	soilPath = "E:\\gye_topo\\whc_800m.tif" 
	ds = gdal.Open(soilPath)
	uvalue = 10 #AWC values are in centimeters, multiply by this to convert to mm
	whc = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype='f8') 
	whc[np.where(whc==100)] = 150 #this represents lakes and basins
	whc[np.where(whc==0)] = 1e-4 #this represents impermeable surfaces
	whc = whc*uvalue
	#uvalue= 200 #for uniform soil water holding capacity
	#whc = uvalue * np.ones([header['nrows'], header['ncols']]) #for the moment creates a uniform soil water holding capacity of 100mm
	return(whc[:,:]) 

def moistureindex(pet, ppt):
	#determines the Thornthwaite moisture index to determine aridity over times series (Thornthwaite 1948) 
	#requires calculation of pet and ppt 
	#MI = ppt-e/e where positive values indicate wetness and negative values indicate dryness
	mi = (ppt-pet)/pet
	return(mi)
#==============================================================================
#=============================Water Model Summary==============================
#==============================================================================

def WaterBalanceModel(BeginYear, EndYear, burnin = 10, **kwargs):
	gcm = kwargs.get('gcm', None)
	rcp = kwargs.get('rcp', None)
	mod = kwargs.get('mod', None)
	if (gcm == None and rcp == None): #if there are no keyword arguments for gcm and rcp, use the PRISM data
		Pm = PRISMextract(BeginYear, EndYear, 'ppt')
		Ta = PRISMextract(BeginYear, EndYear, 'tmean')
		Tdmean = PRISMextract(BeginYear, EndYear, 'tdmean')
		tmin = PRISMextract(BeginYear, EndYear,'tmin')
		tmax = PRISMextract(BeginYear, EndYear, 'tmax')
	else:
		Pm = GCMextract(BeginYear, EndYear, 'ppt', rcp, mod)
		Ta = MeanTemp(BeginYear, EndYear, rcp, mod)
		Tdmean = GCMextract(BeginYear, EndYear, 'tmin',rcp ,mod) #assume tmin is the same as dew point for future scenarios
		tmin = GCMextract(BeginYear,EndYear, 'tmin', rcp, mod)
		tmax = GCMextract(BeginYear,EndYear, 'tmax', rcp, mod)
	swhc = SoilWHC()
	a,s,el = Topoextract()
	VPsat, Vpact, VPD = VapPD(Tdmean, Ta)
	Fm = Meltdata_create(Ta)
	pack = Pack(Fm, Pm)
	Wm, melt, rain = Waterinput(Fm, Pm, pack)
	HL = HeatLI(s,a)
	pet = PET(Ta, HL)
	soilm, aet ,runoff= DeltaSoilCalc(swhc, Wm, pet)
	soilmout = []; aetout = []; petout = []; packout = []; runoffout = []; mi = []; arid = []; vpd = [];
	i = 0 + (burnin*12)
	for year in range(BeginYear+burnin, EndYear+1): #looping through years of interest
		for month in range(1,13): 
			soilmi = GridData(year,month,soilm[i]) 
			aeti = GridData(year,month,aet[i]) 
			peti = GridData(year,month,pet[i]) 
			packi = GridData(year,month,pack[i])
			runoffi = GridData(year,month, runoff[i])
			mii = GridData(year, month, moistureindex(pet[i],Pm[i].data))
			aridi = GridData(year, month, pet[i]/Pm[i].data)
			vpdi = GridData(year, month, VPD[i])
			i += 1
			soilmout.append(soilmi) 
			aetout.append(aeti)
			petout.append(peti) 
			packout.append(packi)
			runoffout.append(runoffi)
			mi.append(mii)
			arid.append(aridi)		
			vpd.append(vpdi)
	return(soilmout, aetout, petout, packout, runoffout, mi, arid, VPD, tmin[burnin*12:], tmax[burnin*12:], Pm[burnin*12:], Ta[burnin*12:])

def aridity(pet, ppt, BeginYear, EndYear):
	datalist= []
	counter=0
	for i in range(BeginYear, EndYear+1):
		for j in range(1,13):
			x= GridData(i,j,pet[counter].data/ppt[counter].data)
			datalist.append(x)
			counter+=1
	return(datalist)
	
def convertWBdata(data, BeginYear, EndYear): #changes the water balance variables into a Grid class
	datalist= []
	counter=0
	for i in range(BeginYear, EndYear+1):
		for j in range(1,13):
			x= GridData(i,j, data[counter])
			datalist.append(x)
			counter+=1
	return(datalist)
	
	#============================================
	#============analysis tools==================
	#============================================
def annualgrid(data,ppt='n'):  #generates the PRISM grids at an annual time step
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
			if ppt =='y':
				x =Annualclimatedata(data[i-1].year,anu_year)
			else:
				x =Annualclimatedata(data[i-1].year,anu_year/counter)
			anu_series.append(x)
			counter = 0
			currentyear = data[i].year
			anu_year = np.zeros(np.shape(data[0].data))
			anu_year +=data[i].data
			counter +=1
		i+=1
	#last iteration
	if ppt == 'y':
		 # if we are interested in ppt per year, we sum the total accumulated ppt and do not divide by 12
		x =Annualclimatedata(data[i-1].year,anu_year)
	else:
		x =Annualclimatedata(data[i-1].year,anu_year/counter)
	anu_series.append(x)
	return(anu_series)  

#======================================================================================
#=============================ANALYSIS FUNCTIONS=======================================
#======================================================================================
def lstfit(data):
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

def temporalgradient(data,p='n'):
	#takes the Pdata and generates the rate of change at the individual cell level
	if p == 'n':
		adata = annualgrid(data)
	else:
		adata = annualgrid(data, 'y')
	dzdt, b0 = lstfit(adata)
	return(dzdt)

def climatemean(data):
	#solves the period mean for plotting purposes
	adata = annualgrid(data)
	nyears = len(adata)
	periodmean = np.zeros(np.shape(data[0].data))
	for i in range(nyears):
		periodmean += adata[i].data
	periodmean = periodmean/nyears
	return(periodmean)
	
def domainmean(data): #returns a time series of the Pdata grid means and detrended series
	Pmean = []
	for i in range(len(data)):
		Pmean.append(np.ma.masked_invalid(data[i].data).mean())
	t = np.arange(len(data))+1
	y = np.array(Pmean)
	b1,b0 = np.polyfit(t, y, 1)
	det_y = y - (b1*t+b0)
	return(y, det_y)
	
def simplemovingavg(ts,lag):
	#determine the lag-month simple moving avg (use the domain mean first)
	run_avg = []
	for i in range(lag, len(ts)):
		run_avg.append(np.mean(ts[i-lag:i]))
	timearray = np.arange(lag,len(ts))
	return(np.array(run_avg))

#======================================================================================
#===============================TIME SERIES PLOT FUNCTIONS=============================
#======================================================================================

def simpleannualplot(data, p='n', tline='y', ma='y', lag=30,l = None, xl=None, yl=None):
	#plots a simple annual time series anomaly of the mean grid area 
	#trend line is plotted by default, can be changed to 'n' to not plot
	adata = annualgrid(data,p)
	t = np.arange(adata[0].year,adata[-1].year+1)
	y, dty = domainmean(adata)
	ax = plt.subplot(1,1,1, axisbg = '0.9',xlabel = xl, ylabel = yl)
	plt.plot(t,y-np.mean(y), label = l)
	if tline=='y':
		b1,b0=np.polyfit(t,y-np.mean(y),1)
		plt.plot(t, t*b1+b0, ls='--',label = 'Trend {0:.2f} $(unit/decade)$'.format(b1*10))
	if ma=='y':
		ts = []
		for i in range(len(adata)):
			ts.append(np.ma.masked_invalid(adata[i].data).mean())
		sma= simplemovingavg(ts,lag)
		plt.plot(t[lag:],sma-np.mean(sma), ls='-.', linewidth =5, label = '%0.f year moving average' %lag)
	plt.grid()
	plt.legend(loc = 'lower right')
	return()

#======================================================================================
#===============================MAP BUILDING FUNCTIONS=================================
#======================================================================================

def hillshade(data,scale=10.0,azdeg=165.0,altdeg=45.0):
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
			ax.add_collection(PatchCollection(ptchs,facecolor ='None',edgecolor='k', linewidths=2))#facecolor=cccol[nshp,:]
	ax.set_xlim(minx,maxx)
	ax.set_ylim(miny,maxy)
	return()

def plotelevation(AOA):
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
	
def plotPRISMgrad(pdata,AOA, p='n'):
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

#======================================================================================
#===============================TIFF WRITE FUNCTIONS===================================
#======================================================================================

def monthlyavg(data, mon):
	mon_data = np.zeros(np.shape(data[0].data))
	counter = 0
	for i in range(len(data)):
		if data[i].month == mon:
			mon_data += data[i].data
			counter += 1
	return(mon_data/counter)

def monthlysummary(data):
	out = []
	for i in range(12):
		out.append(monthlyavg(data,i+1))
	return(out)

def monthlydatalist(datalist):
	out = []
	for i in range(len(datalist)):
		out.append(monthlysummary(datalist[i]))
	return(out)
	
def writetifdata(data, list, workspace,gcm = 'n'):
	mdata = monthlydatalist(data)
	head = Headerextract(gcm)
	for i in range(len(mdata)):
		for j in range(12):
			path = workspace + '\\' 
			name = list[i] + str(j+1) +'.tif'
			Tiffwrite(mdata[i][j], head['ncols'], head['nrows'], head['csize'],head['yul'], head['xul'], path, name)

def ExportData(data,varname,workspace):
	header = Headerextract()
	ny = header['nrows']
	nx = header['ncols']
	cellsize = header['csize']
	yul = header['yul']
	xll = header['xul']
	path = workspace + '\\' + varname + '\\'
	for i in range(len(data)):
		filename = varname + '_' + str(data[i].year) +'_'+ str(data[i].month)+'.tif'
		out = data[i].data
		Tiffwrite(out, nx,ny, cellsize,yul,xll, path, filename)
	return()

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


#=============MAIN=======================
#gather data
varnames =  ['soilm', 'aet', 'pet', 'pack', 'runoff', 'mi', 'arid', 'vpd']
gcmlist = ['CanESM2', 'CCSM4', 'CESM1-BGC','CESM1-CAM5', 'CMCC-CM', 'CNRM-CM5', 'HadGEM2-AO', 'HadGEM2-CC', 'HadGEM2-ES']
rcplist = ['historical']
BeginYear = 1950
EndYear = 2005
burnin = 5
#write the data
#side note that HadGEM2-ES did not have data for 12-2005, so 11-2005 was duplicated to make it work
for mod in gcmlist:
	for rcp in rcplist:
		data = WaterBalanceModel(BeginYear,EndYear, burnin, gcm='y', rcp=rcp, mod=mod)
		for i in varnames:
			#workspace = "E:\\CMIP5\\GCM\\" + mod + "\\rcp" + str(rcp) + "\\wb\\" 
			workspace = "E:\\CMIP5\\GCM\\" + mod + "\\historical" + "\\wb\\" 
			varname = i 
			ind = varnames.index(i)
			ExportData(data[ind], varname, workspace)

#some strange issue with vpd...
#fixed? Not yet 04082014
'''
#renaming convention
import os
for year in range(1950,2006):
    for month in range(11,-1,-1):
        oname = 'vpd_'+str(year)+'_'+str(month)+'.tif'
        nname = 'vpd_'+str(year)+'_'+str(month+1)+'.tif'
        os.rename(oname,nname)
'''

