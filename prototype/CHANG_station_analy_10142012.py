# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:51:37 2012

@author: Tony
"""

import numpy as np
import matplotlib.pyplot as pyplot
from scipy import stats
import pylab
from pylab import *

def data_plot(data,station,startyear, endyear, month, factor):
    #input station of interest, start and end year, and climate measurement
    
    factor_array = []
    time_array = []
    for i in range(numpoints):
        if (data[i].station == station and data[i].year>=startyear and data[i].year <= endyear and data[i].month == month):
            temp = getattr(data[i], factor)            
            factor_array.append(temp) #create array of factors
            time_array.append(data[i].time)
    gradient, intercept, r_value, p_value, std_err = stats.linregress(time_array,factor_array)
    print "Gradient and intercept", gradient, intercept
    print "R-squared", r_value**2
    print "p-value", p_value
    m = polyfit(time_array, factor_array,1)
    yp = polyval(m, time_array)
    fig = pyplot.figure()
    pyplot.title("month " + str(month) + " "+ factor + " plot from " + str(startyear) + " - " + str(endyear) + " for " + station)
    pyplot.ylabel(factor)
    pyplot.xlabel('time')
    pyplot.plot(time_array, factor_array, marker='o')
    if (factor == "ppt"):
        fit_label = 'slope ({0:.3f}) in/year'.format(gradient)
    else:
        fit_label = 'slope ({0:.3f}) F/year'.format(gradient)
    pyplot.plot(time_array,yp, color = 'red', linestyle= '--',label = fit_label)
    pyplot.legend(loc='lower right')    
    return fig
    
def plot_yearlyavg(data,station,startyear, endyear, factor):
    #input station of interest, start and end year, and climate measurement
    if (station == "all"): #if station request is for all stations
        for sc in range(1,stationcode):
            factor_array = []
            time_array = []
            year_array =[]
            for i in range(numpoints):
                if (data[i].stationcode == sc and data[i].year>=startyear and data[i].year <= endyear):
                    year_array.append(data[i].year)
                    temp = getattr(data[i], factor)            
                    factor_array.append(temp) #create array of factors
                    time_array.append(data[i].time)
                    stationname = data[i].station
            x = []
            y = []
            for currentyear in range(startyear, endyear):
                count = 0
                factoravg = 0
                for i in range(len(year_array)):
                    if (year_array[i] == currentyear):
                        if (factor_array[i] != "None"):
                            factoravg += factor_array[i]
                            count += 1
                    if (count==0):
                        break
                    else:
                        y.append(factoravg/count)  
                        x.append(currentyear)
            gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            #print "Gradient and intercept", gradient, intercept
            #print "R-squared", r_value**2
            #print "p-value", p_value
            m = polyfit(x,y,1)
            yp = polyval(m, x)
            fig = pyplot.figure(sc)
            pyplot.subplot(211+sc)
            pyplot.title(factor + " plot from " + str(startyear) + " - " + str(endyear) + " for " + stationname)
            pyplot.ylabel(factor)
            pyplot.xlabel('time')
            pyplot.plot(x,y, marker='o')
            if (factor == "ppt"):
                fit_label = 'slope ({0:.3f}) in/year'.format(gradient)
            else:
                fit_label = 'slope ({0:.3f}) F/year'.format(gradient)
            pyplot.plot(x,yp, color = 'red', linestyle= '--',label = fit_label)
            pyplot.legend(loc='lower right')    
        return fig
        
    else:    
        factor_array = []
        time_array = []
        year_array =[]
        for i in range(numpoints):
            if (data[i].station == station and data[i].year>=startyear and data[i].year <= endyear):
                year_array.append(data[i].year)
                temp = getattr(data[i], factor)            
                factor_array.append(temp) #create array of factors
                time_array.append(data[i].time)
        x = []
        y = []
        for currentyear in range(startyear, endyear):
            count = 0
            factoravg = 0     
            for i in range(len(year_array)):
                if (year_array[i] == currentyear):
                    if (factor_array[i] != "None" or factor_array[i] != ""):
                        factoravg += factor_array[i]
                        count += 1
            if (count != 0):
                y.append(factoravg/count) 
                x.append(currentyear)
                        
        gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        print "Gradient and intercept", gradient, intercept
        print "R-squared", r_value**2
        print "p-value", p_value
        m = polyfit(x,y,1)
        yp = polyval(m, x)
        fig = pyplot.figure()
        pyplot.title(factor + " plot from " + str(startyear) + " - " + str(endyear) + " for " + station)
        pyplot.ylabel(factor)
        pyplot.xlabel('time')
        pyplot.plot(x,y, marker='o')
        if (factor == "ppt"):
            fit_label = 'slope ({0:.3f}) in/year'.format(gradient)
        else:
            fit_label = 'slope ({0:.3f}) F/year'.format(gradient)
        pyplot.plot(x,yp, color = 'red', linestyle= '--',label = fit_label)
        pyplot.legend(loc='lower right')    
        return fig
    
numpoints= 8788 #number of data points in the weather data file

class WData(object):
    #initialize function to construct weather data class
    def __init__(self, stationcode = None,station=None, year=None, month=None, time = None,elev=None, ppt=None, tmax=None, tmin=None):
        self.stationcode = stationcode        
        self.station = station
        self.year = year
        self.month = month
        self.time = time
        self.elev = elev
        self.ppt = ppt
        self.tmax = tmax
        self.tmin = tmin
        
data = [] #List to store all data object
wdata = 'C:\\CHANG\\Climate_Models\\Station_data\\Analysis\\stationdataformattedpython_10162012.csv' 
readfile = open(wdata, 'r')
a = readfile.readline()

for i in range(numpoints):
      a = readfile.readline()
      b = a.split()
      temp = b[0]
      temp = temp.split(",", 9)
      stationcode = int(temp[0])
      station = temp[1]
      year = int(temp[2])
      elev = int(temp[3])
      month = int(temp[4])      
      time = eval(temp[5])
      if (temp[6] == ""):
          ppt = None
      else:
          ppt = eval(temp[6])
      if (temp[7] == ""):
          tmax = None   
      else:
          tmax = eval(temp[7])
      if (temp[8] == ""):
          tmin = None
      else:
          tmin = eval(temp[8])     
      x = WData(stationcode, station, year, month, time, elev, ppt, tmax, tmin)
      data.append(x)     
      
      
    
