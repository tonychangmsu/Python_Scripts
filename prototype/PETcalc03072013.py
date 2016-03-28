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
#import os
#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import mpl
from pylab import *
import time
import matplotlib.animation as animation

#Initialization
"""Required variables to have for calculations
    Ta=         #mean monthly temperature
    Pm=         #mean monthly precipitation
    slope=      #slope of grid cell
    aspect=     #aspect of grid cell
    lat=        #latitude of grid cell
    sd=         #solar declination angle at noon on the 15th day of the month
    soilm=      #soil moisture values

Boundary conditions for site extent are
xmin = -110.948
xmax = -110.425
ymin = 43.537
ymax = 44.133

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
    filename = "D:\\CHANG\\Climate_Models\\US_PRISM_800m\\Uncompressed\\800m_tiff\\tmin\\PRISM800m_tmin1895_1.tif" 
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
    workspace = "D:\\CHANG\\Climate_Models\\US_PRISM_800m\\Uncompressed\\800m_tiff\\"+ var +"\\"    
    for searchyear in range(BeginYear, EndYear+1): #looping through years of interest
        for filenum in range(1,13): #does not consider the annual mean filenum (#14)
            filename = workspace + "PRISM800m_" + var + str(searchyear) + "_" + str(filenum) + ".tif"
            readfile =  gdal.Open(filename)
            data = np.array(readfile.GetRasterBand(1).ReadAsArray())
            x = PRISMData(searchyear,filenum,data) #Create instance of PRISMData object
            Pdata.append(x) 
            readfile = None #close file
    return(Pdata)
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
    aspectPath = "D:\\CHANG\\GIS_Data\\DEM\\TIFF\\asp_gye_800m_NN.tif"
    slopePath = "D:\\CHANG\\GIS_Data\\DEM\\TIFF\\slope_gye800m.tif"
    elevPath = "D:\\CHANG\\GIS_Data\\DEM\\TIFF\\dem_gye800m1.tif"    
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
    '''    
    Fm = np.ma.masked_array(Ta, Ta<=0) #first test if Ta <= 0
    Fm.fill_value = 0
    Fm = Fm.filled()  
    Fm = np.ma.masked_array(Fm, Fm>=6) #second test if Ta >= 6
    Fm.fill_value = 1
    Fm = Fm.filled()
    Fm = 0.167*Fm #multiply entire matix by 0.167 (although the 1 values are to be maintained)
    Fm = np.ma.masked_array(Fm, Fm==0.167) #test if Ta is 0.167 and reassign to 1
    Fm.fill_value = 1 #all values between Ta>=0 and Ta<=6 equal to 0.167*Ta
    Fm = Fm.filled()
    '''
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
    
def DeltaSoilCalc(soilmax,Wm,PET):  #calculate the soil moisture
    soilm = []
    for i in range(len(PET)):    
        if i == 0:
            #s = np.minimum(soilmax, (Wm[i]-PET[i]))
            #s[s<0] = 0
            soilm.append(soilmax)
        else:
            dsoil = soilm[i-1] * (1- np.exp((Wm[i]- PET[i])/soilmax))
            smonth = Wm[i] + dsoil #this is the water lost for the month
            AET = (np.minimum(PET[i], smonth)) #if PET is smaller than ET, than take that value (because we can't lose more than we have)            
            soilmoisture = Wm[i] - AET +soilm[i-1] #water for the month is the water that came in, what came out, plus what was before          
            soilmoisture = (np.minimum(soilmoisture,soilmax))
            soilm.append(soilmoisture) #the culmulative soil moisture for the time step
    return(soilm, AET)

def SoilWHC():
    header = Headerextract()
    whc = 100 * np.ones([header['nrows'], header['ncols']]) #for the moment creates a uniform soil water holding capacity of 100mm
    return (whc)
    
    

"""=============================================================================
========================MAIN====================================================
============================================================================="""

#test data
#slope = linspace(0,(pi/2),7, endpoint=True)
#aspect = linspace(0,2*pi, 360, endpoint=True) #the fold aspect only works from 0 - 180
#lat = linspace(43.537,44.133,90, endpoint=True)
#lat = 44.133
#Defining monthly melt factor, a function of monthly temperature
BeginYear = 2005
EndYear = 2010
Pm = Prec(BeginYear,EndYear)
#Ta = MeanTemp(BeginYear, EndYear)
Ta = PRISMextract(BeginYear,EndYear, 'tmean')
Tdmean = PRISMextract(BeginYear, EndYear, 'tdmean')
VPsat,VPact,VPD = VapPD(Tdmean, Ta)
Fm = Meltdata_create(Ta)
pack = Pack(Fm,Pm)
Wm,melt,rain = Waterinput(Fm,Pm,pack)
a,s,e = Topoextract()
HL = HeatLI(s,a)
pt = PET(Ta,HL)
swhc = SoilWHC()
soilm, AET = DeltaSoilCalc(swhc,Wm, pt)
'''
#visualize
def IPlot(p):
    fig = plt.figure()
    #fig, axes = plt.subplots(3,4)
    im = plt.imshow(p, cmap = 'cool') 
    for i in range(len(p)):
        plt.subplot(3,4,i+1)
        #axes[i].subplot(...
        norm = mpl.colors.Normalize(vmin = 0, vmax =100)
        im = plt.imshow(p[i], cmap = 'cool') 
        im.set_norm(norm)
        plt.colorbar(im)
        #plt.show()
        plt.title('month' + str(Ta[i].month))
    fig.show()
    return()
'''

def p(soilm, i):
    return (soilm[i])


def updatefig(*args):
    global i
    i += 1
    im.set_array(p(soilm,i))
    return im,
    
fig = plt.figure()
i=0
im = plt.imshow(p(soilm,i), cmap = 'cool')
ani = animation.FuncAnimation(fig, updatefig, interval = len(soilm), blit=True)
plt.show()

'''

H = HeatLI(s,a)

#visulizations
figure(1)
for i in range(len(H)):
    subplot(len(H), 1, i)
    ylabel('HI(slope=' + str(math.degrees(slope[i])) +")" )
    xlabel('aspect')
    plot(degrees(aspect),H[i])
  
    
'''
