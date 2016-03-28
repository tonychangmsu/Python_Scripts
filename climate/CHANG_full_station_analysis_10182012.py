# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 13:33:58 2012

@author: Tony
"""
import numpy as np
from scipy import stats
import csv
import matplotlib.pyplot as pyplot

#n = 556411 num of data poitns
n = 49

def mmtemp(data,ind,month,year):
    tempmin = 9999
    tempmax = -9999    
    for i in range(n-1):
        if (data[i].index == ind and data[i].year == year and data[i].month== month):
            if (data[i].tavg < tempmin):
                tempmin = data[i].tavg
            if (data[i].tavg > tempmax):
                tempmax = data[i].tavg
    if (tempmin == 9999):
            tempmin = None
    if (tempmax == -9999):
            tempmax = None
    return(tempmin, tempmax)
            
def data_plot(data,station,startyear, month, endyear):
    #input station of interest, start and end year, and climate measurement
    
    factor_array = []
    time_array = []
    for i in range(n):
        if (data[i].station == station and data[i].year>=startyear and data[i].year <= endyear and data[i].month):
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
    
class WData(object):
    #initialize function to construct weather data class
    def __init__(self, index= None,stationcode = None,station=None, elev=None, typ = None, lat = None, lon = None, year=None, month=None, day = None, tavg=None):
        self.index =index        
        self.stationcode = stationcode        
        self.station = station
        self.elev = elev
        self.typ = typ
        self.lat = lat
        self.lon = lon
        self.year = year
        self.month = month
        self.day = day
        self.tavg = tavg

"""------------------------Main code body-------------------------"""

#n = 556411 num of data poitns

dataline = [] #List to store all data object

with open('C:\\CHANG\\Climate_Models\\Station_data\\Formatted Data\\air_temp_data_test.csv' , 'rb') as csvfile:
    reader = csv.reader(csvfile, dialect = 'excel', delimiter=',')
    for row in reader:
        dataline.append(row)
        
data = []
for i in range(1,n):
      index = int(dataline[i][0])
      station = dataline[i][1]
      stationcode = int(dataline[i][2])
      elev = float(dataline[i][3])
      typ = dataline[i][4]          
      lat = float(dataline[i][5])
      lon = float(dataline[i][6])
      time = dataline[i][8].split("/") #split up date string       
      year = int(time[2])    
      month = int(time[0])
      day = int(time[1])
      tavg = float(dataline[i][9])
      
      x = WData(index, stationcode, station, elev,typ, lat, lon, year, month, day, tavg)
      data.append(x)
      