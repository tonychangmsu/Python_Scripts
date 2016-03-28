'''
Title: 		LANDSAT_data_test.py
Date: 		Created 2016.02.03

Author: 	Tony Chang
Abstract: 	Tests the LANDSAT_tools.py library to observe how the data structure can be manipulated and processed using the enhanced wetness difference index defined as the difference in tasseled wap wetness for two or more dates to detect red-attack stands of lodgepole pine. A decrease in TCWet over time (negative EWDI value) is the best overall indicator of conifer mortality (Collins and Woodcock 1996) 

Should note that this data is already calculated as surface reflectance?

TC_wet = 0.2626*B1+0.2141*B2+0.0926*B3+0.0656*B4-0.7629*B5-0.5388*B7

defined by Huang et al 2002 use landsat images from 1999-2008 for path 38, row 29 to represent core of GYE WBP. Late summer and early fall (ranging from Aug. 9 to Oct. 6) images were chosen to minimize the effects of snow, snowmelt, and understory green-up on TCWet. All images were geometrically registered to a base Landsat image (Sept. 15,1999 ETM+) to within 0.5-pixel root mean squared error since fires were also a problem in detection of disturbance they were filtered out of the analysis. Perimeters of wildfires in the study area were obtained from three sources: US Forest Service Region 1 Fire History Layer (Smail and Tanke 2008), Yellowstone National Park Fire History Layer (Spatial Analysis Center 2005), and the federal Landscape Fire and Resource Management Planning Tools Project (LANDFIRE) Rapid Refresh product, which is derived from MODIS satellite imagery (LANDFIRE 2007). These three fire history data sets were rasterized to 30-m pixels and combined so that a fire recorded in any layer resulted in a fire in the combined raster. Fire perimeters were separately calculated for each individual year, 1999–2008. Cumulative fire history masks were created for each year of imagery (i.e., the 2005 fire mask included fires from 1999, 2000, 2001, 2002, 2003, 2004, and 2005, whereas the 2004 mask did not include 2005 fires).

Table 1. Landsat imagery used in this study
Date Sensor Notes
Sept. 15, 1999 ETM+ Geometric base image
Sept. 20, 2001 ETM+ Radiometric base image
Sept. 23, 2002 ETM+
Oct. 4, 2003 TM
Oct. 6, 2004 TM
Sept. 7, 2005 TM
Aug. 9, 2006 TM
Sept. 21, 2007 ETM+ SLC off
Aug. 22, 2008 ETM+ SLC off
'''

#for now let's just test the transform for the wetness index
import sys
sys.path.append(r'K:\NASA_data\scripts') #add directory for geotools_LT script
import numpy as np
import matplotlib.pyplot as plt
import osgeo.gdal as gdal
import os
import geotool_LT as gt
import LANDSAT_tools as landtools

#==================================MAIN================================================
if __name__ == '__main__':
	pi = 38 #range from 37 to 39
	ri = 29 #range from 28 to 30
	p = np.arange(37,40)
	r = np.arange(28,31)
	begin_year = 2001 #this can increase to the year 1997...
	end_year = 2002 #work with a single year so we can test out the masker
	years = np.arange(begin_year, end_year)
	#use zip here
	#each tile has a bulk order directory
	band = 1
	all_tiles = []
	r_dirs = []
	all_dirs = []
	for year in range(begin_year, end_year):
		for pi in p[:3]:
			tiles = []
			for ri in r[:3]:
				directory_name = r'p%sr%s'%(pi,ri)
				wd = r'G:\NASA_remote_data\Landsat5\%s'%(directory_name)
				scene_dirs = os.listdir(wd) 
				order_name = [s for s in scene_dirs if "Bulk Order" in s] #get the name of the bulk order directory 
				w_path = r"%s\%s\l4-5 tm" %(wd, order_name[0])#assumes only one directory
				#here we should just collect the directories that exist with LT in front
				#get for the year
				dir_list, tif_list = landtools.get_data_list(w_path, year)
				metadata = landtools.get_metadata(dir_list[0])
				bands = landtools.get_bands(dir_list[0])
				r_dirs.append(dir_list[0])
				'''
				tiles.append(bands)
				if ri == r[2]:
					m = landtools.mosaic(tiles[0][band], tiles[1][band])
					m = landtools.mosaic(m, tiles[2][band])
				all_tiles.append(m)
				'''
			all_dirs.append(r_dirs)

#z = landtools.mosaic(all_tiles[0], all_tiles[1])
#z = landtools.mosaic(z, all_tiles[2])
#there are issues when we mosaic East and West. North and South seem to be from the same acquisition date...
#especially with path 30...
#looking at the data, we can see that the paths tend to be taken on the same day, but the rows are not 
#what if we tried between the jd of 151-273

'''
for the year 2000.
		37 		jd = 6
		38 		jd = 29 
		39 		jd = 180/149 

year 2001
		37 		jd = 152/184
		38 		jd = 63/127 
		39 		jd = 86/182
'''
#the only solution for this may be to consider each of the scenes individually, since we have a different date?
#then determine the Date of Disturbance and generate a mosaic of that....


#in any case, I need to correct the data before I mosaic it, I need to use the individual metadata files to
#transform the DN into reflectance and then apply the mosaic method. 
#plt.imshow(z.ReadAsArray(), cmap = 'viridis') 
#plt.show()


#try to get two data sets and mosaic them...
#wetness =  tcWetTransform(data)

#image processing stuff
#http://nbviewer.ipython.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/color_image_processing.ipynb
# may need to first transform the Landsat 5 TM DN data to Landsat 7 ETM+ DN data?
# images are still in DN so they need to be changed to top of atmosphere (ToA) reflectance
#http://www.yale.edu/ceo/Documentation/Landsat_DN_to_Reflectance.pdf
# need to gather the metadata
# now turn to radiance values
#meta data found  
# reflectance calculated
# calculated the tasseled cap transformation via #http://ibis.colostate.edu/webcontent/ws/coloradoview/tutorialsdownloads/co_rs_tutorial10.pdf

'''
from skimage import io, exposure
#img432 = np.dstack((data['B4'], data['B3'], data['B2']))
img432 = np.dstack((ref_data['B4'], ref_data['B3'], ref_data['B2'])) #false color
img321 = np.dstack((ref_data['B3'], ref_data['B2'], ref_data['B1'])) #natural color
img742 = np.dstack((ref_data['B7'], ref_data['B4'], ref_data['B2'])) #natural-like color
img745 = np.dstack((ref_data['B7'], ref_data['B4'], ref_data['B5'])) #natural-like color (penetrates atmospheric haze)
#band combinations information http://web.pdx.edu/~emch/ip1/bandcombinations.html
def color_image_show(img):
	"""
	Show image
	Input:
	img - 3D array of uint16 type
	title - string
	"""
	fig = plt.figure(figsize=(10, 10))
	fig.set_facecolor('white')
	plt.imshow(img/65535)
	plt.show()

fig = plt.figure(figsize=(20, 16))
plt.imshow(img742)

#print the histogram of the colors
fig = plt.figure(figsize=(10, 7))
fig.set_facecolor('white')

for color, channel in zip('rgb', np.rollaxis(img432, axis=-1)):
    counts, centers = exposure.histogram(channel)
    plt.plot(centers[1::], counts[1::], color=color)
plt.show()

#plot equalized image
img432_ha = np.empty(img432.shape, dtype='uint16')
lims = [(0,120),(0,120),(0,120)]
for lim, channel in zip(lims, range(3)):
    img432_ha[:, :, channel] = exposure.rescale_intensity(img432[:, :, channel], lim)

color_image_show(img432_ha)
'''



