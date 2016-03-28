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
from osgeo import gdal
#import gdal
#from gdalconst import *
#import os
#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import mpl

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
    workspace = "D:\\CHANG\\Climate_Models\\PRISM\\tmin\\Uncompressed\\"

    BeginYear = 1895
    filenum = 1
    var = "tmin"                # variable of interest (tmax, tmin, ppt, tdmean)
    PRISMExtent = [-125.02083333333, 24.0625, -66.47916757, 49.9375]
    AOA = [-112.436, 42.252, -108.263, 46.182]      #xmin, ymin, xmax, ymax

    minx = AOA[0] 
    miny = AOA[1]
    maxx = AOA[2]
    maxy = AOA[3]
    
    Pgrid = workspace + "us_" + var + "_" + str(BeginYear) + ".0" + str(filenum) #uncompressed PRISM filename

    readfile = open(Pgrid, 'r')
    a = readfile.readline()
    temp = a.split()
    ncols = int(temp[1])        #Define number of columns
    a = readfile.readline()
    temp = a.split()
    nrows = int(temp[1])        #Define number of rows
    a = readfile.readline()
    temp = a.split()
    xllcorner = float(temp[1])  #Define xll corner
    a = readfile.readline()
    temp = a.split()
    yllcorner = float(temp[1])  #Define yll corner
    a = readfile.readline()
    temp = a.split()
    cellsize  = float(temp[1])  #Define cellsize
    a = readfile.readline()
    temp = a.split()
    NODATA_value  = temp[1]     #Define NoData value
    
    yulcorner = PRISMExtent[1]+(cellsize*nrows)
    
    xstart = int((minx - PRISMExtent[0])/cellsize)    #first x-extent index
    xend = xstart + int((maxx-minx)/cellsize)       #end x-extent index

    ystart = int((yulcorner - maxy)/cellsize)         #first y-extent index
    yend = ystart + int((maxy-miny)/cellsize)       # end of y-extent index
    
    newyulcorner = yulcorner - (ystart*cellsize)
    newxllcorner = PRISMExtent[0] + (xstart*cellsize)
    
    addmatrix = []
    for y_pos in range(0, nrows+1):    #slopy code here to get the number of rows and cols..
        line = readfile.readline()
        datarow = line.split()
        if (y_pos >= ystart and y_pos <= yend):
            newrow = datarow[xstart:(xend+1)]
            addmatrix.append(newrow)
    newcols = len(addmatrix[0]) #define new column length
    newrows = len(addmatrix)    #define new row length
    readfile.close()
    header = [newrows,newcols,newxllcorner,newyulcorner,cellsize,NODATA_value,xstart,xend,ystart,yend,nrows,ncols]    
    return(header)
"============================================================================="

#PRISM data extract function    
def PRISMextract(BeginYear,EndYear,var):        
    header = Headerextract()
    Pdata = [] #List to store all PRISMData object
    #declare the header variables    
    nrows = header[10]
    xstart = header[6]
    xend = header[7]
    ystart = header[8]
    yend = header[9]
    
    workspace = "D:\\CHANG\\Climate_Models\\PRISM\\" + var + "\\Uncompressed\\"
    for searchyear in range(BeginYear, EndYear+1): #looping through years of interest
        '''for filenum in range(1, 15):    #range is to value 14 represents the annual average
            if filenum == 13:
                continue                #month 13 does not exist, skip to the next iteration
            elif filenum < 10:'''#uncomment this code if annual averages are desired
        for filenum in range(1,13):
            if filenum<10:
                Psource = workspace + "us_" + var + "_" + str(searchyear) + ".0" + str(filenum)
            else:
                Psource = workspace + "us_" + var + "_" + str(searchyear) + "." + str(filenum)
            readfile =  open (Psource,'r')
            nhead = 6                   #First 6 lines of the header to be removed
            for z in range(nhead):      #Strip out header
                readfile.readline() #unused lines
            
            addmatrix = []              #List to store PRISM ascii data
            for y_pos in range(0, nrows+1):
                line = readfile.readline()
                datarow = line.split()
                if (y_pos >= ystart and y_pos <= yend):
                    newrow = datarow[xstart:(xend+1)]
                    addmatrix.append(newrow)
            addmatrix = np.array(addmatrix, dtype='i') #changes addmatrix list into array for statistical analysis
            x = PRISMData(searchyear,filenum,addmatrix) #Create instance of PRISMData object
            Pdata.append(x) 
            readfile.close()
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
        storearray = PRISMData(Tmax[i].year, Tmax[i].month, ((Tmax[i].data/100. + Tmin[i].data/100.)/2)) #Definition of Tmean from Daly 2012
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
        VPactstorearray = PRISMData(Tmean[i].year, Tmean[i].month, a*np.exp(((b*(Tdmean[i].data/100.))/((Tdmean[i].data/100.)+c))))
        VPDmeanstorearray = PRISMData(Tmean[i].year, Tmean[i].month, VPsatstorearray.data-VPactstorearray.data)         
        VPsat.append(VPsatstorearray)
        VPact.append(VPactstorearray)
        VPDmean.append(VPDmeanstorearray)
    return(VPsat,VPact,VPDmean)
    
#coordinate locator function based on data matrix indices
def Cretrieve(xindex,yindex):
    header = Headerextract()
    xmin = header[2]    
    ymin = header[3]
    csize = header[4]
    lat = ymin-yindex*csize
    lon = xmin+xindex*csize
    coor = [lat,lon]
    return(coor)

def Gridcreate():
    header = Headerextract()
    lat_list = []
    lon_list = []
    nrows = header[0]
    ncols = header[1]    
    xmin = header[2]
    ymin = header[3]
    csize = header[4]
    for ystep in range(nrows):
        latstep = ymin-ystep*csize
        lat_list.append(latstep)
    for xstep in range(ncols):
        lonstep = xmin+xstep*csize
        lon_list.append(lonstep)
    lat = array([lat_list,]*ncols).transpose()
    lon = array([lon_list,]*nrows)
    return(lat,lon)

def Prec(BeginYear,EndYear):
    ppt = PRISMextract(BeginYear,EndYear,'ppt')
    for i in range(len(ppt)):
        ppt[i].data = ppt[i].data/100.
    return(ppt)

def Topoextract():
    aspectPath = "D:\\CHANG\\GIS_Data\\DEM\\TIFF\\aspect4kmbi1.tif"
    slopePath = "D:\\CHANG\\GIS_Data\\DEM\\TIFF\\slope4km1.tif"
    elevPath = "D:\\CHANG\\GIS_Data\\DEM\\TIFF\\GCSdem4km.tif"    
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
    return (Fm)

def Meltdata_create(Ta):
    Fm_list = []
    for i in range(len(Ta)):
        Fm = Meltfactor(Ta[i].data)
        Fm_list.append(Fm)
    return(Fm_list)
    
def Pack(Fm, Pm):    
    pack = []    
    packprev = 0 #initial pack condition assumed to be 0
    for i in range(len(Fm)):
        packprev=(((1-Fm[i])**2 * Pm[i].data) + ((1-Fm[i]) *packprev))
        pack.append(packprev)
    return(pack)
    
def Waterinput(Fm, Pm, pack):
    wm_list = []    
    for i in range(len(Fm)):    
        rain_m = Fm[i]*Pm[i].data
        snow_m = (1-Fm[i])*Pm[i].data
        if i == 0:
            melt_m = Fm[i] * (snow_m)  #pack assumed to be 0 at first month
        else:
            melt_m = Fm[i] * (snow_m + pack[i-1])
        wm = rain_m + melt_m
        wm_list.append(wm)
    return(wm_list)
    
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
    
def PETcalc(Ta,month,hl):    
    ea = 0.611*(np.exp((17.27*Ta)/(Ta+237.3))) #saturation vapour pressure
    dayspmonth = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    days = np.ones(np.shape(Ta))*dayspmonth[month-1] #create grid of shape(Ta) filled with number of days of given month   
    sd = Solardec()
    lat,lon = Gridcreate()
    dl = Daylength(sd,lat)
    PET = 29.8*days*dl[month-1]*hl*(ea/(Ta+273.2))
    return(PET)

def Soilcalc(soilmax,Wm,PET,soilm):  #calculate the soil moisture
    soilWHC = [soilmax, (Wm-PET)+soilm] #soilmax is the soil water holding capacity in the top 200cm of the soil profile    
    return(min(soilmax))


"""=============================================================================
========================MAIN====================================================
============================================================================="""

#test data
#slope = linspace(0,(pi/2),7, endpoint=True)
#aspect = linspace(0,2*pi, 360, endpoint=True) #the fold aspect only works from 0 - 180
#lat = linspace(43.537,44.133,90, endpoint=True)
#lat = 44.133
#Defining monthly melt factor, a function of monthly temperature
'''
H=[]
for i in range(len(slope)):
    H.append(HeatLI(lat,slope[i],Aspectf(aspect)))
'''
BeginYear = 2000
EndYear = 2003
Pm = Prec(BeginYear,EndYear)
Ta = MeanTemp(BeginYear, EndYear)
Tdmean = PRISMextract(BeginYear, EndYear, 'tdmean')
VPsat,VPact,VPD = VapPD(Tdmean, Ta)
Fm = Meltdata_create(Ta)
pack = Pack(Fm,Pm)
Wm = Waterinput(Fm,Pm,pack)
a,s,e = Topoextract()
HL = HeatLI(s,a)
p = []
for i in range(12):
    p.append(PETcalc(Ta[i].data,Ta[i].month,HL))
p = np.array(p)

#visualize
def IPlot(p):
    figure()
    for i in range(len(p)):
        plt.subplot(3,4,i+1)
        #norm = mpl.colors.Normalize(vmin = 0, vmax = 135)
        m = plt.imshow(p[i], cmap = 'cool') 
        #m.set_norm(norm)
        plt.colorbar(m)
        plt.show()
        plt.title('month' + str(Ta[i].month))
    return()

#H = HeatLI(s,a)

'''
#visulizations
figure(1)
for i in range(len(H)):
    subplot(len(H), 1, i)
    ylabel('HI(slope=' + str(math.degrees(slope[i])) +")" )
    xlabel('aspect')
    plot(degrees(aspect),H[i])
'''
    
    
