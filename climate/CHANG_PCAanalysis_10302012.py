# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:39:44 2012

@author: Tony
"""

import numpy as np
from numpy import mean,cov,double, cumsum, dot,array, rank
import numpy.linalg as la
import matplotlib.pyplot as pyplot
from scipy.linalg import svd
import matplotlib.mlab as mlab
from matplotlib import cm as cm

def PCAdataformat(data, numstations, factor, startyear, endyear):
    nummonths = 12 
    PCAlist = []
    index = []
    for i in range(1,nummonths+1):
        temp = [] #store each array of data and factor type
        indextemp = []
        for j in range(len(data)-1):                #iterate through all the data
            if (data[j].year>=startyear and data[j].year <= endyear and data[j].month == i): #check if the data is within the specified years
                temp.append(getattr(data[j],factor))
                indextemp.append(j)
        PCAlist.append(temp)
        index.append(indextemp)
    return(PCAlist,index) 