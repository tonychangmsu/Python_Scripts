"""Correlogram construction test
Programmer: Tony Chang
Course: BIOE 504 Quantitative Biology
Institution: Montana State University Bozeman
Instructor: Dr. Daniel Goodman

Description: This program will input a time series array and develop a correlogram
to evalute the autoregression type.

"""
import numpy
from numpy import *
import scipy
from scipy import *
import matplotlib


def RandomNumGen(numDig,iniSeed,numIt):
    #input: number of desired digits, an initial decimal seed value, and number of iterations
    #output: numIt random numbers generated based on Von Neumann algorithm    
    randomNum = []    
    if numDig%2 != 0:
        numDig += 1  #must be even
    for i in range (numIt):
        iniSeed = iniSeed * (10**numDig)
        randomNum.append(int(iniSeed))        
        newNum = iniSeed**2
        newNum = floor(newNum/(10**(numDig/2))) #remove last two digits
        newNum = newNum / (10**numDig) 
        iniSeed = newNum - floor(newNum) #remove first two digits 
    clf() #clear plot figure
    hist(randomNum, bins = 25)     
    return(randomNum)
    
def Means(data):
    #input: time series array dataset
    #output: returns mean of dataset
    mu = 0
    for i in range(len(data)):
        mu = mu + data[i]
    return (mu/len(data))

def Variance(data):
    #input: time series array dataset
    #output: returns variance of dataset
    mu_hat = Means(data)
    n = len(data)
    sigmasq = 0
    for i in range(n):
        sigmasq = (data[i]**2) + sigmasq 
    sigmasq = (sigmasq/n) - mu_hat**2
    return(sigmasq)

def StandDev(data):
    #input: time series array dataset
    #output: returns standard deviation of dataset
    return (sqrt(Variance(data)))

def Covar(datax, datay):
    #input: time series array dataset and array of corresponding time values
    #output: returns covariance array
    cov = 0
    n = len(datax)
    #Numeric solution to cov = 1/n*(datax[i]-Means(datax))*(datay[i]-Means(datay))
    for i in range(len(datax)):
        cov = cov + datax[i]*datay[i]
    cov = (cov/n - ((Means(datax)*Means(datay))))
    return(cov)

def CorrMat(data):
    #input: time series array dataset and array of corresponding time values
    #output: returns correlation matrix
    n = len(data) #number of columns
    corrM = zeros((n,n))
    for i in range(n):
        for j in range (n):
            corrM[i,j] = Covar(data[i],data[j])/(StandDev(data[i])*StandDev(data[j]))
    return(corrM)

def CovMat(data):
    #input: time series array of n columns and m rows
    #output: returns the covariance matrix 
    n = len(data) #number of columns
    covM = zeros((n,n))
    for i in range(n):
        for j in range (n):
            covM[i,j] = Covar(data[i],data[j])
    return(covM)
    
def Correllogram(data):
    import matplotlib.pyplot as plt
    #input: time series array dataset
    #output: returns correlogram of lag n 
    n = len(data)
    cor_array = zeros(n)
    lag = zeros(n)
    
    for i in range(n):      #first loop through each column vector of matrix
        #for k in range(len(data[i))):   #second loop for each element of vector 
        cor_array[i] = Covar(data[:n-i],data[i:])
        lag[i] = i
        """
        c = (data[:n-i],data[i:])
        print 'Lag ', lag[i].astype(int), ' covariants' 
        print(CovMat(c))
        print 'correlation matrix:'
        print(CorrMat(c))
        print ''
        """
    y = cor_array[:n-1]
    x = lag[:n-1]
    plt.plot(x,y)
    return(x,y)

def ttest(x,y):
    import scipy
    from scipy import stats
    #input: array of x(time) values and DesignMat class y values 
    #output: performs a standard linear regression analysis and t-test to determine significance
            #return will be a StatsMat class with slope, intercept, r-square, p-value, and std-err
    class StatsMat(object):
        def __init__(self, slope, intercept, r_squared, p_value, std_err):
            self.slope = slope
            self.intercept = intercept
            self.r_squared = r_squared
            self.p_value = p_value
            self.std_err = std_err
    Sdata = []   
    for i in range(len(y)):    
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y[i].cVal)
        rsquare = r_value**2
        s = StatsMat(slope, intercept, rsquare, p_value, std_err)
        Sdata.append(s)
    return(Sdata)
    
def ConvertMat(Pdata,Sdata):
    #input: Pdata class with all attributes of cellsize, columns num, and rows
    #output: Sdata in numpy matrix format for geospatial transform
    n = Pdata[0].ncols
    m = Pdata[0].nrows
    TranData = zeros((n,m))
    i = 0
    for c in range(n):
        for r in range(m):
            TranData[c][r] = Sdata[i].p_value
            i += 1
    return(TranData)

"""
#---------Write ESRI raster file
def WriteGrid(Pdata,Sdata,filename):
    import arcpy
    gnrows = Pdata[0].PRval.shape[0]
    gxllcorner = Pdata[0].xll
    gyllcorner = Pdata[0].yul - (gnrows*Pdata[0].csize)
    corner = arcpy.Point(gxllcorner, gyllcorner)
    gcellsize = Pdata[0].csize
    gNODATA = int(Pdata[0].NODATA)

    myRaster = arcpy.NumPyArrayToRaster(Sdata, corner, gcellsize, gcellsize, gNODATA)
    projection = "GEOGCS['GCS_WGS_1972',DATUM['D_WGS_1972',SPHEROID['WGS_1972',6378135.0,298.26]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]]"
    arcpy.DefineProjection_management(myRaster,projection)
    myRaster.save("C:/CHANG/PRISM/PRISM_Analysis/" + filename)  #change file name here
    return []
"""

    
    
    
    
