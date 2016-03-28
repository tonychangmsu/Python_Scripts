#Title: GYE_adult_wbp_filter.py
#Abstract: get the adult wbp trees for analysis
#Author: Tony Chang
#Date: 10/16/2014

import pandas as pd
import numpy as np
def filterSpecies(fiadata,cde):
#provide fia plot data and species code desired and returns queried fia data
	return(fiadata[(fiadata['SPCD'] == cde)])

def getAbsence(fiadata, out_cde):
#provide fia plot data and species code to be excluded
	return(fiadata[(fiadata['SPCD'] != out_cde)])
	
def linkCN(fiadata, plot_table):
#links the fiadata to the reference plot_table by the PLT_CN attribute
	return(plot_table[plot_table.CN.isin(fiadata.PLT_CN)])

def filterArea(fiadata, AOA):
#provide fia plot data and AOA as a list = [xmin, ymin, xmax, ymax]
	return(fiadata[(fiadata.LON > AOA[0]) & (fiadata.LON < AOA[2]) & (fiadata.LAT > AOA[1]) & (fiadata.LAT < AOA[3])])


gyetrees = pd.read_csv('E:\\FIA\\GYETREES.csv')
whitebark_code = 101
adultDBH = 7.87
wbp = filterSpecies(gyetrees, whitebark_code)
adultwbp = wbp[wbp.DIA>=adultDBH]

plots = pd.read_csv('E:\\FIA\\PLOT.csv') #these plots are not limited to GYE (they represent all plots?)
nonwbp = getAbsence(gyetrees, whitebark_code)

lwbp = linkCN(adultwbp, plots)
lnonwbp = linkCN(nonwbp, plots)

xmax = -108.263; xmin = -112.436; ymin = 42.252; ymax = 46.182 # GYE bounds
AOA = [xmin, ymin, xmax, ymax] #specify the bounds for the FIA data

wbp_adult_gye = filterArea(lwbp, AOA)
nonwbp_gye = filterArea(lnonwbp,AOA)

#add the presence column
wbp_adult_gye['presence'] = pd.Series(np.ones(len(wbp_adult_gye)), index = wbp_adult_gye.index)
nonwbp_gye['presence'] = pd.Series(np.zeros(len(nonwbp_gye)), index = nonwbp_gye.index)

#combine the presences and absences
pres = wbp_adult_gye[['presence', 'LAT', 'LON', 'ELEV']]
abse = nonwbp_gye[['presence', 'LAT', 'LON', 'ELEV']]
wbpdata = pres.append(abse)
wbpdata.to_csv('E:\\FIA\\WBP_ADULT_PA.CSV') #written out 10/16/2014 @t.chang