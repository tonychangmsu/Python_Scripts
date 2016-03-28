# -*- coding: utf-8 -*-
"""
Created on Mon Mar 04 18:14:32 2013

@author: tony.chang
"""

import numpy 
from numpy import *
from osgeo import osr
from osgeo import gdal, gdal_array
from osgeo.gdalconst import GDT_Float32

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
    return("filebuilt!\n")

BeginYear = 1895
EndYear = 2010
filenum = 1
var = "tmean"                # variable of interest (tmax, tmin, ppt, tdmean)
workspace = "D:\\CHANG\\Climate_Models\\US_PRISM_800m\\uncompressed\\" + var + "\\"
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
readfile.close()

yulcorner = PRISMExtent[1]+(cellsize*nrows)

xstart = int((AOA[0] - PRISMExtent[0])/cellsize)    #first x-extent index
xend = xstart + int((AOA[2]-AOA[0])/cellsize)       #end x-extent index

ystart = int((yulcorner - AOA[3])/cellsize)         #first y-extent index
yend = ystart + int((AOA[3]-AOA[1])/cellsize)       # end of y-extent index

for searchyear in range(BeginYear, EndYear+1): #looping through years of interest
    annualdata = []
    for filenum in range(1, 15):
        addmatrix = []              #List to store PRISM ascii data
        if filenum == 13:
            continue                #month 13 does not exist, skip to the next iteration
        elif filenum < 10:
            Psource = workspace + "us_" + var + "_" + str(searchyear) + ".0" + str(filenum)
        else:
            Psource = workspace + "us_" + var + "_" + str(searchyear) + "." + str(filenum)
        readfile =  open (Psource,'r')
        nhead = 6                   #First 6 lines of the header to be removed
        for z in range(nhead):      #Strip out header
            a = readfile.readline()
        for y_pos in range(0, nrows+1):
            line = readfile.readline()
            datarow = line.split()
            if (y_pos >= ystart and y_pos <= yend):
               newrow = datarow[xstart:(xend+1)]
               addmatrix.append(newrow)
        newcols = len(addmatrix[0]) #define new column length
        newrows = len(addmatrix)    #define new row length
        newyulcorner = yulcorner - (ystart*cellsize)
        newxllcorner = PRISMExtent[0] + (xstart*cellsize)
        addmatrix = numpy.array(addmatrix) #changes addmatrix list into array for statistical analysis
        addmatrix = (addmatrix.astype(float32))/100
        annualdata.append(addmatrix)
        tiffname = "PRISM800m_"+ var + str(searchyear) +"_"+ str(filenum) + ".tif"
        PRISMtiffwrite(addmatrix,var,tiffname,newcols,newrows,cellsize,newyulcorner,newxllcorner,1)
