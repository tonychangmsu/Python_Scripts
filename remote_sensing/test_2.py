import sys
sys.path.append(r'K:\NASA_data\scripts') #add directory for geotools_LT script
import numpy as np
import matplotlib.pyplot as plt
import osgeo.gdal as gdal
import os
import geotool_LT as gt
import LANDSAT_tools as landtools

#i'm actually not sure if i have surface reflectance product or something completely different....

ref = gdal.Open(r'G:\NASA_remote_data\Landsat5\p38r28\Bulk Order 441935\L4-5 TM\LT50380282003213PAC02\LT50380282003213PAC02_B1.tif')

l_source = r'G:\NASA_remote_data\Landsat5\p38r28\Bulk Order 441935\L4-5 TM\LT50380282003213PAC02'
DN_data = []
ref_data = []
keybands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
for i in keybands:
	for j in f_names:
		if (i in j):
			#get tiff
			ds = gdal.Open(r'%s\%s'%(l_source, j)).ReadAsArray()
			#transform the ds of DN5 to DN7
			#check if the user input a mask
			ds = landtools.LT5ToLT7(ds, i)
			#if len(mask) != 0:
			#	ds = np.ma.array(ds, mask = mask)
			DN_data.append(ds)	
			#calculate the reflectance
			#rad = DN5ToRad(ds, metadata, i) #changed to use the DN7 transform
			rad = landtools.DN7ToRad(ds, i)
			ref = landtools.radToRef(rad, metadata, i) #this reflectance algorithm only works for DN7 not DN5
			ref[ref<0] = 0 # max sure all values are positive
			ref_data.append(ref)
			ds = None #close data
			continue
			
d_dict = dict(zip(keybands, DN_data)) # create dict of the data
r_dict = dict(zip(keybands, ref_data))

data = landtools.getLSdata(lsource, 

meta = landtools.get_metadata(lsource)
test = ref.ReadAsArray()
test_lt7 = landtools.LT5ToLT7(test, 1)
rad = landtools.DN7ToRad(test_lt7, 1)
ref = landtools.radToRef(rad, meta, 1)

