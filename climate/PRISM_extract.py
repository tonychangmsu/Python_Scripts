import numpy
from numpy import *

workspace = "C:\\CHANG\\PRISM\\tmax\\Uncompressed\\"

BeginYear = 1895
EndYear = 2011
filenum = 1
var = "tmax" # variable of interest (tmax, tmin, ppt, tdmean)
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

class PRISMData(object):
    #initialize function to construct PRISMdata class
    def __init__(self, year=None, month=None, ncols=None, nrows=None, xll=None, yll=None, csize=None, NODATA=None, PRval=None):
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
        self.ncols = ncols
        self.nrows = nrows
        self.xll = xll
        self.yll = yll
        self.csize = csize
        self.NODATA = NODATA
        self.PRval = PRval
        
Pdata = [] #List to store all PRISMData object

for searchyear in range(BeginYear, EndYear): #looping through years of interest
    for filenum in range(1, 15):
        addmatrix = []  #List to store PRISM ascii data
        if filenum == 13:
            continue #month 13 does not exist, skip to the next iteration
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
        newyllcorner = AOA[1]
        newxllcorner = AOA[0]
        addmatrix = numpy.array(addmatrix) #changes addmatrix list into array for statistical analysis
        x = PRISMData(searchyear,filenum, newcols, newrows, newxllcorner, newyllcorner, cellsize, NODATA_value, addmatrix) #Create instance of PRISMData object
        Pdata.append(x) 
        
