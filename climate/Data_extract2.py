import glob, sys, os, decimal, numpy
from numpy import *

workspace = "C:\\CHANG\\PRISM\\tmax\\Uncompressed\\"

BeginYear = 1895
EndYear = 1897
filenum = 1 # varies between 1-12 corresponding to month
var = "tmax" # variable of interest (tmax, tmin, ppt)
AOA = [-112.436, 42.252, -108.263, 46.182] #xmin, ymin, xmax, ymax

minx = AOA[0] 
miny = AOA[1]
maxx = AOA[2]
maxy = AOA[3]

Pgrid = workspace + "us_" + var + "_" + str(BeginYear) + ".0" + str(filenum)

readfile = open(Pgrid, 'r')
a = readfile.readline()
temp = a.split()
ncols = int(temp[1]) #Define number of columns
a = readfile.readline()
temp = a.split()
nrows = int(temp[1])  #Define number of rows
a = readfile.readline()
temp = a.split()
xllcorner = float(temp[1])   #Define xll corner
a = readfile.readline()
temp = a.split()
yllcorner = float(temp[1])   #Define yll corner
a = readfile.readline()
temp = a.split()
cellsize  = float(temp[1])   #Define cellsize
a = readfile.readline()
temp = a.split()
NODATA_value  = temp[1]      #Define NoData value
readfile.close()

yulcorner = yllcorner+(cellsize*nrows)

xstart = int((AOA[0] - xllcorner)/cellsize) #first x-extent index
xend = xstart + int((AOA[2]-AOA[0])/cellsize) #end x-extent index

ystart = int((yulcorner - AOA[3])/cellsize) #first y-extent index
yend = ystart + int((AOA[3]-AOA[1])/cellsize) # end of y-extent index

analysisList = []

for year in range(BeginYear, EndYear): #looping through years of interest
    datamatrix = []
    Pgrid = workspace + "us_" + var + "_" + str(year) + ".0" + str(filenum)
    readfile =  open (Pgrid,'r')
    nhead = 6                   #First 6 lines of the header to be removed
    for z in range(nhead):      #Strip out header
        a = readfile.readline()
    for y_pos in range(0, nrows+1):
        line = readfile.readline()
        datarow = line.split()
        if (y_pos >= ystart and y_pos <= yend):
           newrow = datarow[xstart:(xend+1)]
           datamatrix.append(newrow)
        
    newcols = len(datamatrix[0]) #define new column length
    newrows = len(datamatrix)    #define new row length
    newyllcorner = AOA[1]
    newxllcorner = AOA[0]
    newheader = [newcols, newrows, newxllcorner, newyllcorner, cellsize, NODATA_value]
    datamatrix = numpy.array(datamatrix) #changes datamatrix list into array for statistical analysis
         
    analysisList.append([year,datamatrix])
    
