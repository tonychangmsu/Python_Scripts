# Uncomment the next two lines if you want to save the animation
#import matplotlib
#matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation
import pandas as pd

class PRISMData(object):
    #initialize function to construct PRISMdata class
    def __init__(self, year=None, month=None, data=None):
        self.year = year
        self.month = month
        if month == 14:         #month 14 in PRISM data represents the mean of the years
            self.season = "ALL"
        elif (month < 3 or month == 12):
            self.season = "Win"
        elif month < 6:
            self.season = "Spr"
        elif month < 9:
            self.season = "Sum"
        else:
            self.season = "Fal"
        self.data = data

class Annualclimatedata(object):
	#annual climate data summary in same form as PRISMData object
	def __init__(self, year=None, data=None):
		self.year = year
		self.data = data

def PRISMextract(BeginYear,EndYear,var): #temporal bounds of data from 1895-2010
    Pdata = [] #List to store all PRISMData object
    if (var == 'dtr'): #if the variable is dtr, then we use a different routine    
        var1='tmin'
        var2='tmax' 
        workspace1 = "D:\\CHANG\\Climate_Models\\US_PRISM_800m\\Uncompressed\\800m_tiff\\"+ var1 +"\\"
        workspace2 = "D:\\CHANG\\Climate_Models\\US_PRISM_800m\\Uncompressed\\800m_tiff\\"+ var2 +"\\"
        for searchyear in range(BeginYear, EndYear+1): #looping through years of interest
            for filenum in range(1,13): #does not consider the annual mean filenum (#14) 
                filename = workspace1 + "PRISM800m_" + var1 + str(searchyear) + "_" + str(filenum) + ".tif"
                readfile =  gdal.Open(filename)
                data1 = np.array(readfile.GetRasterBand(1).ReadAsArray())
                readfile = None #close file
                filename2 = workspace2 + "PRISM800m_" + var2 + str(searchyear) + "_" + str(filenum) + ".tif"
                readfile2 =  gdal.Open(filename2)
                data2 = np.array(readfile2.GetRasterBand(1).ReadAsArray())
                readfile2 = None
                x = PRISMData(searchyear,filenum,data2-data1) #Create instance of PRISMData object
                Pdata.append(x) 
    else:
        workspace = "D:\\CHANG\\Climate_Models\\US_PRISM_800m\\Uncompressed\\800m_tiff\\"+ var +"\\"
        for searchyear in range(BeginYear, EndYear+1): #looping through years of interest
            for filenum in range(1,13): #does not consider the annual mean filenum (#14)
                filename = workspace + "PRISM800m_" + var + str(searchyear) + "_" + str(filenum) + ".tif"
                readfile =  gdal.Open(filename)
                data = np.array(readfile.GetRasterBand(1).ReadAsArray())
                x = PRISMData(searchyear,filenum,data) #Create instance of PRISMData object
                Pdata.append(x) 
                readfile = None #close file	
    return(Pdata)

def Topoextract():
    aspectPath = "D:\\CHANG\\GIS_Data\\DEM\\TIFF\\asp_gye_800m_NN.tif"
    slopePath = "D:\\CHANG\\GIS_Data\\DEM\\TIFF\\slope_gye800m.tif"
    elevPath = "D:\\CHANG\\GIS_Data\\DEM\\TIFF\\dem_gye800m1.tif"   
    ds = gdal.Open(aspectPath)
    aspect = np.array(ds.GetRasterBand(1).ReadAsArray())
    ds = gdal.Open(slopePath)
    slope = np.array(ds.GetRasterBand(1).ReadAsArray())
    ds = gdal.Open(elevPath)
    elev = np.array(ds.GetRasterBand(1).ReadAsArray())
    ds = None #close files
    return(aspect,slope,elev)   

def annualgrid(data):  #generates the PRISM grids at an annual time step
	numyears = int(len(data)/12)
	anu_year = np.zeros(np.shape(data[0].data))
	anu_series = []
	by = data[0].year
	ey = data[-1].year
	currentyear =by
	i=0
	counter =0
	while (i<len(data)):
		if (currentyear == data[i].year):
			anu_year+=data[i].data
			counter +=1
		elif (currentyear != data[i].year):
			x =Annualclimatedata(data[i-1].year,anu_year/counter)
			anu_series.append(x)
			counter = 0
			currentyear = data[i].year
			anu_year = np.zeros(np.shape(data[0].data))
		i+=1
	#last iteration
	x =Annualclimatedata(data[i-1].year,anu_year/counter)
	anu_series.append(x)
	return(anu_series)  
	
# Sent for figure
font = {'size'   : 9}
plt.rc('font', **font)

# Setup figure and subplots
f0 = plt.figure(num = 0, figsize = (12, 8))#, dpi = 100)
f0.suptitle("Oscillation decay", fontsize=12)

ax05 = plt.subplot2grid((5,4),(0,0), rowspan=4, colspan=4)
ax06 = plt.subplot2grid((5,4),(4,0), colspan=4)
#tight_layout()

# Set titles of subplots
ax05.set_title('sample image')
'''
# set y-limits
ax01.set_ylim(0,2)
ax02.set_ylim(-6,6)
ax03.set_ylim(-0,5)
ax04.set_ylim(-10,10)

# sex x-limits
ax01.set_xlim(0,5.0)
ax02.set_xlim(0,5.0)
ax03.set_xlim(0,5.0)
ax04.set_xlim(0,5.0)

# Turn on grids
ax01.grid(True)
ax02.grid(True)
ax03.grid(True)
ax06.grid(True)

# set label names
ax01.set_xlabel("x")
ax01.set_ylabel("py")
ax02.set_xlabel("t")
ax02.set_ylabel("vy")
ax03.set_xlabel("t")
ax03.set_ylabel("py")
ax04.set_ylabel("vy")

# Data Placeholders
yp1=np.zeros(0)
yv1=np.zeros(0)
yp2=np.zeros(0)
yv2=np.zeros(0)
'''
t=np.zeros(0)

#Set of random images
images = np.random.randn(250,10,10)
imean = []
for j in range (len(images)):
	imean.append(np.mean(images[j]))
imean=np.array(imean)
j = np.arange(len(imean))
b1,b0 = np.polyfit(j,imean,1)
isim = b1*j+b0
# set plots
'''
p011, = ax01.plot(t,yp1,'b-', label="yp1")
p012, = ax01.plot(t,yp2,'g-', label="yp2")

p021, = ax02.plot(t,yv1,'b-', label="yv1")
p022, = ax02.plot(t,yv2,'g-', label="yv2")

p031, = ax03.plot(t,yp1,'b-', label="yp1")
p032, = ax04.plot(t,yv1,'g-', label="yv1")
'''
im = ax05.imshow(images[0])
p04, = ax06.plot(j,imean)
p05, = ax06.plot(j,isim, '--',color='red', label ="trend line")

# set lagends
'''
ax01.legend([p011,p012], [p011.get_label(),p012.get_label()])
ax02.legend([p021,p022], [p021.get_label(),p022.get_label()])
ax03.legend([p031,p032], [p031.get_label(),p032.get_label()])
'''
# Data Update
i=0
def updateData(self):
	
	global t
	global images
	global i
	global imean

	i += 1
	
	im.set_data(images[i])
	
	p04.set_data(j[:i],imean[:i])
	p05.set_data(j[:i],isim[:i])

	return ()

# interval: draw new frame every 'interval' ms
# frames: number of frames to draw
simulation = animation.FuncAnimation(f0, updateData, blit=False, frames=250, interval=1, repeat=False)

# Uncomment the next line if you want to save the animation
#simulation.save(filename='sim.mp4',fps=30,dpi=300)

plt.show()
