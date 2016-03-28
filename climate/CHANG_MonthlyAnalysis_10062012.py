# -*- coding: utf-8 -*-
import numpy 
from numpy import *
"""
Created on Sat Oct 06 15:24:41 2012

@author: Tony
"""
"""
def Means(Pdata):
    import numpy 
    from numpy import *

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
    xbar = xbar/df
    ybar = ybar/df
    return(xbar, ybar, df)
"""
def MonthGradient(month, Pdata, startyear, endyear):


    #initialize zero arrays 
    n = Pdata[0].nrows
    m = Pdata[0].ncols
    
    xbar = zeros((n,m))
    ybar = zeros((n,m))
    Sxx = zeros((n,m))
    Sxy = zeros((n,m))
    df = 0
    
    for i in range(len(Pdata)): #solve for the means
        if (Pdata[i].year >= startyear and Pdata[i].year <= endyear and Pdata[i].month == month):
            xi = (ones((n,m)) * (Pdata[i].year)) #each month getting (1/12) of year value
            yi = (Pdata[i].PRval.astype(float)/100)
            xbar = xbar + xi
            ybar = ybar + yi
            df = df+1
    xbar = xbar/df
    ybar = ybar/df
    
    for i in range(len(Pdata)):
        if (Pdata[i].year >= startyear and Pdata[i].year <= endyear and Pdata[i].month == month): #consider within the year range and month needed
            xi = (ones((n,m)) * (Pdata[i].year))
            yi = (Pdata[i].PRval.astype(float)/100)
            Sxx = Sxx +((xi-xbar)*(xi-xbar))
            Sxy = Sxy +((xi-xbar)*(yi-ybar))
    monthGrad = Sxy/Sxx
    monthGrad = monthGrad.astype(float32)
    return (monthGrad)
"""
def SlopeMat(sea, Pdata, startyear, endyear):
    import numpy 
    from numpy import *

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
    return (slopeMat)
"""