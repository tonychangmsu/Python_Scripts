#!/usr/local/env python
from mpl_toolkits.basemap import Basemap
import numpy as np #used to preform simple math functions on data
from netCDF4 import Dataset
import numpy.ma as M
#decide which file to open
import matplotlib.pyplot as pyplot #used to build contour and wind barbs plots
from pylab import *


#the below example produce a KML friendly image which includes a basemap
fig = pyplot.figure()
grid = np.zeros((181,360))
nlat = 90
slat = -90
wlong = 0
elong = 359
ax = pyplot.axes((0,0,1,1))  #  Key line for a KML friendly image
m = Basemap(projection='cyl',resolution='c',llcrnrlon=wlong,llcrnrlat=slat,urcrnrlon=elong,urcrnrlat=nlat,fix_aspect=False)
# in the line above the kwarg fix_aspect=False must be included to produce a KML friendly image
x,y = m(*np.meshgrid(range(wlong,elong+1),range(slat,nlat+1)))
pyplot.plot = m.contourf(x,y,grid)
m.drawcoastlines() #draw coastlines
m.drawmapboundary() #draw a line around the map region
m.fillcontinents(color='0.8', lake_color=None, ax=None, zorder=2) #fill in continents with color (gray)
m.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0]) #draw parallels
m.drawmeridians(np.arange(0,359,30),labels=[0,0,0,1]) #draw meridians
fig.savefig('test-kml.png')

'''
#below is an example that produces a KML friendly image without a basemap
fig = figure()
# create some data to use for the plot
dt = 0.001
t = arange(0.0, 10.0, dt)
r = exp(-t[:1000]/0.05)               # impulse response
x = randn(len(t))
s = convolve(x,r)[:len(x)]*dt  # colored noise
ax = axes((0,0,1,1))
pyplot.plot(t, s)
xlabel('time (s)')
ylabel('current (nA)')
title('Gaussian colored noise')
show()
fig.savefig('test1-kml.png')
'''