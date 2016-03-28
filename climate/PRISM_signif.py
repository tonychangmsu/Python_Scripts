import numpy 
from numpy import *

workspace = "C:\\CHANG\\Climate_Models\\PRISM\\tmin\\Uncompressed\\"

BeginYear = 1895
EndYear = 2011
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
readfile.close()

yulcorner = PRISMExtent[1]+(cellsize*nrows)

xstart = int((AOA[0] - PRISMExtent[0])/cellsize)    #first x-extent index
xend = xstart + int((AOA[2]-AOA[0])/cellsize)       #end x-extent index

ystart = int((yulcorner - AOA[3])/cellsize)         #first y-extent index
yend = ystart + int((AOA[3]-AOA[1])/cellsize)       # end of y-extent index

class PRISMData(object):
    #initialize function to construct PRISMdata class
    def __init__(self, year=None, month=None, ncols=None, nrows=None, xll=None, yul=None, csize=None, NODATA=None, PRval=None):
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
        self.yul = yul
        self.csize = csize
        self.NODATA = NODATA
        self.PRval = PRval
        
Pdata = [] #List to store all PRISMData object

for searchyear in range(BeginYear, EndYear+1): #looping through years of interest
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
        x = PRISMData(searchyear,filenum, newcols, newrows, newxllcorner, newyulcorner, cellsize, NODATA_value, addmatrix) #Create instance of PRISMData object
        Pdata.append(x) 
        

#PRISM data analysis
""" To be called after PRISM_extract, will write functions for the
PRISM_extract code once I'm done with this section.

Data is stored in a list of classes called PData in the following format:

PData[index].year
PData[index].month
PData[index].ncols
PData[index].nrows
PData[index].xll
PData[index].yul
PData[index].csize
PData[index].NODATA
PData[index].PRval

PRval contains a NumPy array of the correponding PRISM data
"""
"""
def z_score(Pdata,mu):
    import scipy
    from scipy import stats
    xbar = Means(Pdata)[0]
    ybar = Means(Pdata)[1]
    n = Means(Pdata)[2]
    df = n-1
    var = UnbiasSTD(Pdata)
    sampVar = var/ math.sqrt(n)
    z = ybar - 
     
def UnbiasSTD(Pdata):
    n = Pdata[0].nrows
    m = Pdata[0].ncols
    STD = zeros((n,m))
    ybar = Means(Pdata)[1]
    N = len(Pdata)
    for i in range(len(Pdata)):
        yi = (Pdata[i].PRval.astype(float)/100)
        STD = (yi - ybar)*(yi - ybar) + SS
    STD = sqrt(SS/(N-1))
    return (STD)
   
def Means(Pdata):
    n = Pdata[0].nrows
    m = Pdata[0].ncols
    xbar = zeros((n,m))
    ybar = zeros((n,m))
    for i in range(len(Pdata)):
        if (Pdata[i].year >= startyear and Pdata[i].year <= endyear and Pdata[i].season == sea):
            if (sea == "ALL"): # case where we consider the full year
                xi = (ones((n,m)) * (Pdata[i].year))
            else:
                xi = (ones((n,m)) * (Pdata[i].year + (Pdata[i].month * (1/12.)))) #each month getting (1/12) of year value
            yi = (Pdata[i].PRval.astype(float)/100)
            xbar = xbar + xi
            ybar = ybar + yi
            df = df+1
            counter = counter + 1
    xbar = xbar/df
    ybar = ybar/df
    return(xbar, ybar, df)

def SlopeMat(sea, Pdata, startyear, endyear):
    #initialize zero arrays 
    counter = 0
    n = Pdata[0].nrows
    m = Pdata[0].ncols
    
    sumy = zeros((n,m)) 
    sumx = zeros((n,m))
    sumxy = zeros((n,m))
    sumxsq = zeros((n,m))

    xbar = Means(Pdata)[0]
    ybar = Means(Pdata)[1]
    df = Means(Pdata)[2]

    Sxx = zeros((n,m))
    Sxy = zeros((n,m))
    Syy = zeros((n,m))
    SST = zeros((n,m))
    SSW = zeros((n,m))
    SSB = zeros((n,m))
    meanbar = (xbar +ybar)/(df*2)
    """
    for i in range(len(Pdata)):
        if (Pdata[i].year >= startyear and Pdata[i].year <= endyear and Pdata[i].season == sea):
            if (sea == "ALL"): # case where we consider the full year
                xi = (ones((n,m)) * (Pdata[i].year))
            else:
                xi = (ones((n,m)) * (Pdata[i].year + (Pdata[i].month * (1/12.)))) #each month getting (1/12) of year value
            yi = (Pdata[i].PRval.astype(float)/100)
            xbar = xbar + xi
            ybar = ybar + yi
            df = df+1
            counter = counter + 1
    xbar = xbar/df
    ybar = ybar/df
    #print (counter)
    """
    for i in range(len(Pdata)):
        if (Pdata[i].year >= startyear and Pdata[i].year <= endyear and Pdata[i].season == sea):
            if (sea == "ALL"):
                xi = (ones((n,m)) * (Pdata[i].year))
            else:
                xi = (ones((n,m)) * (Pdata[i].year + (Pdata[i].month * (1/12.))))
            yi = (Pdata[i].PRval.astype(float)/100)
            Sxx = Sxx +((xi-xbar)*(xi-xbar))
            Sxy = Sxy +((xi-xbar)*(yi-ybar))
            SST = SST +((yi-meanbar)*(yi-meanbar))+((xi-meanbar)*(xi-meanbar)) #n-1 df
            SSW = SSW +((xi-xbar)*(xi-xbar))+((yi-ybar)*(yi-ybar)) #m*(n-1) df
    SSB = (df*(xbar-meanbar)*(xbar-meanbar)) + (df*(ybar-meanbar)*(ybar-meanbar)) #m-1 df
    slopeMat = Sxy/Sxx
    slopeMat = slopeMat.astype(float32)
    Fstat= (SSB/1.)/(SSW/(df-1)) #SSB/(m-1) / SSW/df-1
    alpha = 0.05 #level of significance
    return (slopeMat,)

"""
#---------Write ESRI raster file
def WriteGrid(gMat,Pdata, filename):
    import arcpy
    gncols = gMat.shape[1]
    gnrows = gMat.shape[0]
    gxllcorner = Pdata[0].xll
    gyllcorner = Pdata[0].yul - (gnrows*Pdata[0].csize)
    corner = arcpy.Point(gxllcorner, gyllcorner)
    gcellsize = Pdata[0].csize
    gNODATA = int(Pdata[0].NODATA)

    myRaster = arcpy.NumPyArrayToRaster(gMat, corner, gcellsize, gcellsize, gNODATA)
    projection = "GEOGCS['GCS_WGS_1972',DATUM['D_WGS_1972',SPHEROID['WGS_1972',6378135.0,298.26]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]]"
    arcpy.DefineProjection_management(myRaster,projection)
    myRaster.save("C:/CHANG/PRISM/PRISM_Analysis/" + filename)  #change file name here
    return []


#Below is attempt to import to GTiff, needs work, possibly version errors 

from osgeo import osr
from osgeo import gdal, gdal_array
from osgeo.gdalconst import GDT_Float32

Ny, Nx = gMat.shape
cwidth = Pdata[0].csize
Yul = Pdata[0].yul
Xul = Pdata[0].xll

fileformat = "GTiff"
driver = gdal.GetDriverByName(fileformat)
geotransform = [Xul, cwidth,0.0, Yul, 0.0, -cwidth]
srs = osr.SpatialReference()
srs.ImportFromEPSG(4322) #4322 coordinate ref# for World Geodetic System 1972
#writename = "C:\CHANG\PRISM\PRISM_Analysis\us_" + var + "_" + str(BeginYear) + "_" + str(EndYear) + "_gradients.tif"
writename = "C:\CHANG\PRISM\PRISM_Analysis\a.tif"
outDs = driver.Create(writename, Nx, Ny, 1, gdal.GDT_Float32)
outDs.SetGeoTransform(geotransform)
outDs.SetProjection(srs.ExportToWkt())
outBand = outDs.GetRasterBand(1)
outBand.SetNoDataValue(-9999)
outBand.WriteArray(gMat, 0, 0)
#gdal_array.BandWriteArray(outBand,gMat)
"""





