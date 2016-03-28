#Title: MPB_Cold_T_test.py
#Author: Tony Chang
#Date: 02.03.2015
#Abstract: Testing the functionality of the MPB_coldT_model.py functions

import MPB_coldT_model as mpb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#main
datafile = 'E:\\MPB_model\\MPB_phenology\\DATA\\temp.txt'
raw_temp = pd.read_csv(datafile, header = None)
T = raw_temp.dropna() #the temperature from field data with the na removed
#note that this data set represents 4 years of data hourly 
#get the daily highs and lows
Tmin = []
Tmax = []

for i in range(0, len(T), 24):
	Tmin.append(np.min(T[i:i+24])[0])
	Tmax.append(np.max(T[i:i+24])[0])
Tmin = np.array(Tmin) 
Tmax = np.array(Tmax)

#want to use only 365 days from the summer
start_index = 212
end_index = 577

Tmin_in = Tmin[start_index:end_index] - np.random.random(365)*10
Tmax_in = Tmax[start_index:end_index]
out = mpb.runModel(Tmin_in, Tmax_in, year = 1990)

plt.rcParams['figure.figsize'] = 14,6
for i in range(len(out['C'])):
	mpb.plotPop(out['C'][i], save='y', n=i)