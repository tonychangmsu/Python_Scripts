# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:51:37 2012

@author: Tony
"""

import numpy as np
from numpy import linspace, ones, convolve
import matplotlib.pyplot as pyplot
from scipy import stats
import pylab
from pylab import *
import csv
import sys

def monthsort(data,startyear, endyear, factor):
    #divides data into monthly groups
    month_data = []
    station_index = []    
    for i in range(len(data)):
        if (data[i].year >=startyear and data[i].year <=endyear):
            for j in range(1,13):
                if (data[i].month == j):
                    temp = getattr(data[i],factor)
                    if (temp != None):
                        month_array.append(temp)
                        station_index.append(data[i].stationcode)
    

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval,window, 'same')#[window_size-1:-(window_size-1)]
    
def plot_data(data,stationcode, startyear, endyear, factor):
    #input station of interest, start and end year, and climate measurement
    factor_array = []
    year_array = []        
    for i in range(numpoints):           
        if (data[i].stationcode == stationcode and data[i].year>=startyear and data[i].year <= endyear):                               
            temp = getattr(data[i],factor)                 
            if (temp != None):                
                factor_array.append(temp) #create array of factors
                year_array.append(data[i].year)
                st = data[i].station
    x = []
    y = []
    for currentyear in range(startyear, endyear+1): #get the yearly averages
        count = 0
        factoravg = 0
        for k in range(len(year_array)):
            if (year_array[k] == currentyear):
                factoravg += factor_array[k]
                count += 1
        if (count != 0  and count >=9):
            y.append(factoravg/count)  
            x.append(currentyear)                         
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y = y-mean(y)    
    m = polyfit(x,y,1)
    yp = polyval(m, x)
    y_av = movingaverage(y,10)
    #y_av[0]=y[0]
    #y_av[len(y_av)-1]=y[len(y)-1]
    fig = pyplot.figure()
    pyplot.title(factor + " plot from " + str(startyear) + " - " + str(endyear) + " for " + st)
    #pyplot.ylabel(factor + " anomaly ($^{o}$F)")
    pyplot.ylabel(factor + " anomaly (in)")
    pyplot.xlabel('time')
    pyplot.grid(b=True, which='major')
    #meanline =([x[0],mean(y)],[x[len(x)-1],mean(y)])
    #pyplot.plot(meanline)
    pyplot.plot(x,y, marker='o')
    if (factor == "ppt"):
        fit_label = 'slope ({0:.3f}) in/year'.format(gradient)
    else:
        fit_label = 'slope ({0:.3f}) $^o$F/year'.format(gradient)
    pyplot.plot(x,yp, color = 'red', linestyle= '--',label = fit_label)
    pyplot.plot(x,y_av, color = 'green', label = '10-yr running mean')
    pyplot.legend(loc='lower right')    
    return fig

        
def full_data_analyze(data,startyear, endyear, factor):
   
    #input station of interest, start and end year, and climate measurement 
    #fig = pyplot.figure()
    factor_array = []
    year_array = []        
    for i in range(numpoints):           
        if (data[i].year>=startyear and data[i].year <= endyear):                               
            temp = getattr(data[i],factor)                 
            if (temp != None):                
                factor_array.append(temp) #create array of factors
                year_array.append(data[i].year)
    x = []
    y = []
    for currentyear in range(startyear, endyear+1): #get the yearly averages
        count = 0
        factoravg = 0
        for k in range(len(year_array)):
            if (year_array[k] == currentyear and factor_array[k] != None):
                factoravg += factor_array[k]
                count += 1
        if (count >= 1):
            y.append(factoravg/count)  
            x.append(currentyear)   
    #gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    #results = [gradient, r_value**2, p_value]
   
    y = y-mean(y)    
    m = polyfit(x,y,1)
    yp = polyval(m, x)
    y_av = movingaverage(y,10)
    #y_av[0]=y[0]
    #y_av[len(y_av)-1]=y[len(y)-1]
    print x
    print y  
    pyplot.title(factor + " plot from " + str(startyear) + " - " + str(endyear) + " for all analyzed stations")
    pyplot.ylabel(factor + " $^{o}$F/year")
    pyplot.xlabel('time')
    pyplot.grid(b=True, which='major')
    #meanline =([x[0],mean(y)],[x[len(x)-1],mean(y)])
    #pyplot.plot(meanline)
    pyplot.plot(x,y, marker='o')
    if (factor == "ppt"):
        #fit_label = 'trend ({0:.3f}) in/year'.format(gradient)
        fit_label = 'trend in/year'
    else:
        fit_label = 'trend {0:.3f} ($^o$F/year)'.format(gradient)
    pyplot.plot(x,yp, color = 'red', linestyle= '--',label = fit_label)
    pyplot.plot(x,y_av, color = 'green', label = '10-yr running mean')
    pyplot.legend(loc='upper right')    
    pyplot.show()
    return [x,y]#results
    
def data_analyze(data,startyear, endyear, factor):
   
    #input station of interest, start and end year, and climate measurement
    results = []  
    #fig = pyplot.figure()
    for j in range(1,numstations+1):
        factor_array = []
        year_array = []        
        for i in range(numpoints):           
            if (data[i].stationcode == j and data[i].year>=startyear and data[i].year <= endyear):                               
                temp = getattr(data[i],factor)                 
                if (temp != None):                
                    factor_array.append(temp) #create array of factors
                    year_array.append(data[i].year)
        x = []
        y = []
        for currentyear in range(startyear, endyear+1): #get the yearly averages
            count = 0
            factoravg = 0
            for k in range(len(year_array)):
                if (year_array[k] == currentyear):
                    factoravg += factor_array[k]
                    count += 1
            if (count != 0  and count >=9):
                y.append(factoravg/count)  
                x.append(currentyear)              
        if (len(factor_array) > 0):
            gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            rsq = r_value**2
            out = statoutput(j, factor, x[0], x[len(x)-1], gradient, rsq, p_value)
            results.append(out)
            m = polyfit(x,y,1)
            yp = polyval(m,x)
            pyplot.plot(x,y)
            pyplot.plot(x,yp, color = 'red', linestyle = '--')
            pyplot.show()
    return results

def data_monthly_analyze(data,startyear, endyear, elev, factor, month):
    #input station of interest, start and end year, and climate measurement 
    #fig = pyplot.figure()
    factor_array = []
    year_array = []        
    for i in range(numpoints):           
        if (data[i].elev> elev and data[i].month == month and data[i].year>=startyear and data[i].year <= endyear):                               
            temp = getattr(data[i],factor)                 
            if (temp != None):                
                factor_array.append(temp) #create array of factors
                year_array.append(data[i].year)
    x = []
    y = []
    for currentyear in range(startyear, endyear+1): #get the yearly averages
        count = 0
        factoravg = 0
        for k in range(len(year_array)):
            if (year_array[k] == currentyear and factor_array[k] != None):
                factoravg += factor_array[k]
                count += 1
        if (count >= 1):
            y.append(factoravg/count)  
            x.append(currentyear)   
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    results = [gradient, r_value**2, p_value]
   
    y = y-mean(y)    
    m = polyfit(x,y,1)
    yp = polyval(m, x)
    y_av = movingaverage(y,10)
    #y_av[0]=y[0]
    #y_av[len(y_av)-1]=y[len(y)-1]
    print results
    fig = pyplot.figure()
    pyplot.title(factor + " plot from " + str(startyear) + " - " + str(endyear) + " for all analyzed stations in month " + str(month) )
    pyplot.ylabel(factor + " $^{o}$F/year")
    pyplot.xlabel('time')
    pyplot.grid(b=True, which='major')
    #meanline =([x[0],mean(y)],[x[len(x)-1],mean(y)])
    #pyplot.plot(meanline)
    pyplot.plot(x,y, marker='o')
    if (factor == "ppt"):
        #fit_label = 'trend ({0:.3f}) in/year'.format(gradient)
        fit_label = 'trend in/year'
    else:
        fit_label = 'trend {0:.3f} ($^o$F/year)'.format(gradient)
    pyplot.plot(x,yp, color = 'red', linestyle= '--',label = fit_label)
    pyplot.plot(x,y_av, color = 'green', label = '10-yr running mean')
    pyplot.legend(loc='upper right')    
    pyplot.show()
    return results
    
numpoints= 20290 #number of data points in the weather data file
numstations = 44

class WData(object):
    #initialize function to construct weather data class
    def __init__(self, stationcode = None,station=None, typ= None, elev = None,lat = None, lon = None, year=None, month=None, time=None, ppt=None, tmax=None, tmin=None, dtr = None):
        self.stationcode = stationcode        
        self.station = station
        self.typ = typ
        self.elev = elev
        self.lat = lat
        self.lon = lon
        self.year = year
        self.month = month
        self.time = time
        self.ppt = ppt
        self.tmax = tmax
        self.tmin = tmin
        self.dtr = dtr

class statoutput(object):
    def __init__(self, stationcode=None, factor=None, startyear=None, endyear=None, gradient=None, rsq=None, pval=None):
        self.stationcode = stationcode
        self.factor = factor
        self.startyear = startyear
        self.endyear = endyear
        self.gradient = gradient
        self.rsq = rsq
        self.pval = pval
        
data = [] #List to store all data object
wdata = 'C:\\CHANG\\Climate_Models\\Station_data\\Formatted Data\\Combined_data_monthlyPYTHON_10202012.csv' 
readfile = open(wdata, 'r')
a = readfile.readline()

for i in range(numpoints):
      a = readfile.readline()
      b = a.split()
      temp = b[0]
      temp = temp.split(",", 12)
      stationcode = int(temp[0])
      station = temp[1]
      typ = temp[2]
      elev = float(temp[3])
      lat = float(temp[4])
      lon = float(temp[5])
      year = int(temp[6])
      month = int(temp[7])
      time = float(temp[8])
      if (temp[9] == ""):
          ppt = None
      else:
          ppt = eval(temp[9])
      if (temp[10] == ""):
          tmax = None   
      else:
          tmax = eval(temp[10])
      if (temp[11] == ""):
          tmin = None
      else:
          tmin = eval(temp[11])     
      if (temp[10] == "" or temp[11] == ""):
          dtr =None
      else:
          dtr = tmax-tmin
      x = WData(stationcode, station,typ, elev,lat, lon, year, month, time, ppt, tmax, tmin, dtr)
      data.append(x)     
"""
result = data_analyze(data,1989,2011,"tmin")
output = []
for i in range(len(result)):
    output.append([result[i].stationcode, result[i].factor,result[i].startyear, result[i].endyear, result[i].gradient, result[i].rsq, result[i].pval])

f = open("output10232012_1989_2011tmin.csv","w")
writer = csv.writer(f, delimiter=',')
writer.writerow( ('station code','tmin', 'startyear', 'endyear', 'gradient', 'rsq', 'pval' ))
for i in range(len(output)):
    writer.writerow(output[i])
f.close()
"""