#	Title: 		QA_check.py
#	Author: 	Tony Chang
#	Date: 		09.25.2015
#	Abstract:	Quality check on the hourly model for TOPOWX

import numpy as np
import matplotlib.pyplot as plt
import gdal as gdal
import netCDF4 as nc

#open the netCDF4 files and plot a few random points then look at the signal

wd = "K:\\NASA_data\\hourly_topowx\\"
year = np.arange(2000, 2006)
xpoint = 200
ypoint = 200
temp = np.array([])
day = np.array([])
year_add = 0
for y in year:
	filename = "%s%s_topowx_hourly_temp.nc"%(wd, y)
	ds = nc.Dataset(filename)
	temp = np.concatenate((temp, (ds.variables['temperature'][:, xpoint, ypoint])))
	day = np.concatenate((day, (ds.variables['time'][:] + year_add)))
	year_add = np.round(ds.variables['time'][-1]) + year_add

plt.rcParams['figure.figsize'] = 20, 10
plt.plot(day, temp)
plt.xlim(0, day[-1])

#looks fairly decent....
