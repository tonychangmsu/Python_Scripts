# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:35:49 2012

@author: Tony
"""

import numpy as np
from numpy import mean,cov,double, cumsum, dot,array, rank
import numpy.linalg as la
import matplotlib.pyplot as pyplot
from scipy.linalg import svd
import matplotlib.mlab as mlab
from matplotlib import cm as cm


csv = "C:\\CHANG\\Climate_Models\\Station_Data\\Analysis\\Extraction\\PRISM_gradCompare10252012.csv"  

r = mlab.csv2rec(csv)

prism_tmax = []
s_tmax = []
prism_tmin = []
s_tmin = []

for i in range(len(r)):
    prism_tmax.append(r[i][6])
    s_tmax.append(r[i][7])
    prism_tmin.append(r[i][8])
    s_tmin.append(r[i][9])
b1= prism_tmax
b2 = s_tmax
b3= prism_tmin
b4 = s_tmin
b5 = [b1,b2]
b6 = [b3, b4]
"""
fig = pyplot.figure()
ax = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax.boxplot(b5)
ax2.boxplot(b6)
"""
pyplot.boxplot(b6)
#pyplot.boxplot(b6)
#pyplot.xticks([1,2],['PRISM Tmax', 'stations Tmax'])
pyplot.xticks([1,2],['PRISM Tmin', 'stations Tmin'])
pyplot.title('PRISM vs weather stations validation')
pyplot.ylabel("trend ($^{o}$F/year)")
pyplot.show()