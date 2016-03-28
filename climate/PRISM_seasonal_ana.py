import numpy 
from numpy import *
'''
workspace = "D:\\CHANG\\PRISM\\tmin\\Uncompressed\\"

BeginYear = 1982
EndYear = 2011
filenum = 1
var = "tmin"                # variable of interest (tmax, tmin, ppt, tdmean)
PRISMExtent = [-125.02083333333, 24.0625, -66.47916757, 49.9375]
AOA = [-112.436, 42.252, -108.263, 46.182]      #xmin, ymin, xmax, ymax

minx = AOA[0] 
miny = AOA[1]
maxx = AOA[2]
maxy = AOA[3]
'''
def SeasonMat(sea, Pdata, startyear, endyear):
    """Analysis of the trend line for specified season of Pdata class"""
    #initialize zero arrays 
    counter = 0
    n = Pdata[0].nrows
    m = Pdata[0].ncols
    df = 0
    sumy = zeros((n,m)) 
    sumx = zeros((n,m))
    sumxy = zeros((n,m))
    sumxsq = zeros((n,m))

    xbar = zeros((n,m))
    ybar = zeros((n,m))

    Sxx = zeros((n,m))
    Sxy = zeros((n,m))
    for i in range(len(Pdata)):
        if (Pdata[i].year >= startyear and Pdata[i].year <= endyear and Pdata[i].season == sea):
            if (sea == "all"): # case where we consider the full year
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
    for i in range(len(Pdata)):
        if (Pdata[i].year >= startyear and Pdata[i].year <= endyear and Pdata[i].season == sea):
            if (sea == "ALL"):
                xi = (ones((n,m)) * (Pdata[i].year))
            else:
                xi = (ones((n,m)) * (Pdata[i].year + (Pdata[i].month * (1/12.))))
            yi = (Pdata[i].PRval.astype(float)/100)
            Sxx = Sxx +((xi-xbar)*(xi-xbar))
            Sxy = Sxy +((xi-xbar)*(yi-ybar))
    gMat = Sxy/Sxx
    gMat = gMat.astype(float32)
    return (gMat)

def monthMat(month, Pdata, startyear, endyear):
    """Analysis of the trend line through time for Pdata class by specified months 1-12"""
    #initialize zero arrays 
    counter = 0
    n = Pdata[0].ncols
    m = Pdata[0].nrows
    df = 0
    sumy = zeros((n,m)) 
    sumx = zeros((n,m))
    sumxy = zeros((n,m))
    sumxsq = zeros((n,m))

    xbar = zeros((n,m))
    ybar = zeros((n,m))

    Sxx = zeros((n,m))
    Sxy = zeros((n,m))
    for i in range(len(Pdata)): #calculate the means
        if (Pdata[i].year >= startyear and Pdata[i].year <= endyear and Pdata[i].month == month):
            xi = (ones((n,m)) * (Pdata[i].year)) #start adding x's
            yi = (Pdata[i].PRval.astype(float)/100) #start adding y's for the given month
            xbar = xbar + xi
            ybar = ybar + yi
            df = df+1
            counter = counter + 1
    xbar = xbar/df
    ybar = ybar/df
    #print (counter)
    for i in range(len(Pdata)): #calculate the trend lines
        if (Pdata[i].year >= startyear and Pdata[i].year <= endyear and Pdata[i].month == month):
            xi = (ones((n,m)) * (Pdata[i].year))
            yi = (Pdata[i].PRval.astype(float)/100)
            Sxx = Sxx +((xi-xbar)*(xi-xbar))
            Sxy = Sxy +((xi-xbar)*(yi-ybar))
    gMat = Sxy/Sxx
    gMat = gMat.astype(float32)
    return (gMat)




