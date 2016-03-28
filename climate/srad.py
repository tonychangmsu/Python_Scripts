#solar radiation #2
# model derived from J.G. Corripio, 2003, Vectorial algebra algorithms for calculating terrain parameters from DEMS and solar radiation modelling in 
# mountainous terrain

import numpy as np
from matplotlib import pyplot as plt
import gdal
from gdalconst import *
import osr

def dem_extract(source = "E:\\gye_topo\\dem_800m.tif"):
	#input dem as tif file
	#since shape of the GCM is different from the PRISM extent, a modified grid is required if gcm parameter is set to 'y'   
	elevPath = source
	ds = gdal.Open(elevPath)
	elev = np.array(ds.GetRasterBand(1).ReadAsArray())
	ds = None #close files
	return(elev)

def solvenormal(data, zo, l):
	#l is the cell size
	#zo is the tuple of initial point
	dza = data[zo[0]+1,zo[1]]-data[zo[0],zo[1]]
	dzb = data[zo[0],zo[1]+1]-data[zo[0],zo[1]]
	dzc = data[zo[0],zo[1]+1]-data[zo[0]+1,zo[1]+1]
	dzd = data[zo[0]+1,zo[1]]-data[zo[0]+1,zo[1]+1]
	a = np.array([l,0,dza])
	b = np.array([0,l,dzb])
	c = np.array([-l, 0, dzc])
	d = np.array([0,-l,dzd])
	n = 0.5*(np.cross(a,b)+np.cross(c,d))
	return(n)

def solveslope(data,zo,l):
	n = solvenormal(data,zo,l)
	#solve for the unit n
	nu = n/(np.dot(n,n)**.5)
	return(np.arccos(nu[2]))

def solveaspect(data,zo,l):# needs a lot of work and thought....
	n = solvenormal(data,zo,l)
	#solve for the unit n
	nu = n/(np.dot(n,n)**.5)
	theta=np.arctan(nu[1]/nu[0]) + np.pi/2
	asp = np.nan
	#find the directions
	if(nu[0] >0 and nu[1] >0): #quadrant 1
		asp = theta
	elif(nu[0] >0 and nu[1]<0): #quadrant 2
		asp = theta + np.pi/2
	elif(nu[0]< 0 and nu[1] < 0): #quadrant 3
		asp = theta + np.pi
	elif(nu[0]< 0 and nu[1] >0): #quadrant 4
		asp = theta + np.pi*(3/2)
	return(asp)

def makeslope(data,l):
	s = np.zeros(np.shape(data))
	for i in range(np.shape(data)[0]-1):
		for j in range(np.shape(data)[1]-1):
			zo = (i,j)
			s[i,j] = solveslope(data,zo,l)
	return(s)
	
def makeaspect(data,l):
	a = np.zeros(np.shape(data))
	for i in range(np.shape(data)[0]-1):
		for j in range(np.shape(data)[1]-1):
			zo = (i,j)
			a[i,j] = solveaspect(data,zo,l)
	return(a)
	
#main 
elev = dem_extract()
l = 800
s = makeslope(elev,l)
a = makeaspect(elev,l)
plt.subplot(211)
plt.imshow(s*180/np.pi); plt.colorbar()
plt.subplot(212)
plt.imshow(a*180/np.pi); plt.colorbar()