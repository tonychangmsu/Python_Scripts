# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:49:00 2012

@author: Tony
"""


import numpy as np
from numpy import mean,cov,double, cumsum, dot,array, rank
import numpy.linalg as la
import matplotlib.pyplot as pyplot
from scipy.linalg import svd
import matplotlib.mlab as mlab
from matplotlib import cm as cm
from scipy import stats
import pylab
from pylab import *


csv = "C:\\CHANG\\Climate_Models\\Station_Data\\Analysis\\Extraction\\exportPRISM_DEM_Grad.csv"  

r = mlab.csv2rec(csv)
x =[]
y = []
for i in range(len(r)-1):
    x.append(r[i][4])
    y.append(r[i][5])
gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)
m = polyfit(x,y,1)
yp = polyval(m, x)
fig = pyplot.figure()
pyplot.title("PRISM Tmin trend plot from 1982 - 2011")
pyplot.ylabel("trend ($^{o}$F/year)")
pyplot.xlabel('Elevation (ft)')
pyplot.grid(b=True, which='major')
    #meanline =([x[0],mean(y)],[x[len(x)-1],mean(y)])
    #pyplot.plot(meanline)
pyplot.scatter(x,y, marker='o')
fit_label = 'slope {0:.3f} ($^o$F/year)/ 1000ft'.format(gradient*1000)
pyplot.plot(x,yp, color = 'red', linestyle= '--',label = fit_label)
pyplot.legend(loc='lower right')    