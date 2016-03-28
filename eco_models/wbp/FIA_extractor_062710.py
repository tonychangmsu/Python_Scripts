"""
Title: Open/parse FIA data
Author: Tony Chang
"""

import numpy as np
from matplotlib import pyplot as plt

#=============================================================================================================

def data_AOA(AOA, plot):
	#returns the data filtered by bounding box AOA
	lat = plot['LAT']
	lon = plot['LON']
	lat_cut = (lat>=AOA[2]) & (lat<=AOA[3])
	lon_cut = (lon>=AOA[0]) & (lon<=AOA[1])
	return(plot[lat_cut & lon_cut])

def get_CN(data):
	cn = []
	for i in range(len(data)):
		cn.append(data[i].decode('utf-8')[1:-1])
	cn = np.array(cn) #messy method....
	return(cn)

def find_CN_match(data1, data2):
	indices = []
	for i in data2:
		check = np.where(data1==i)[0]
		if len(check) != 0:
			indices.append(check)
	return(np.array(indices))
	#match CNs and returns the indices for data1
	
#=============================================================================================================
	
#=======================================
workspace = 'E:\\Downloads\\'
seedname = 'SEEDLING.CSV'
seedfile = '%s%s' %(workspace,seedname)
seed = np.genfromtxt(seedfile, delimiter = ',', names = True, dtype = None)

plotname = 'PLOT.CSV'
plotfile = '%s%s' %(workspace,plotname)
'''
plotdtype = dtype([('CN', '<U15'), ('SRV_CN', '<U15'), ('CTY_CN', '<U14'), ('PREV_PLT_CN', '<U15'), ('INVYR', '<i4'), ('STATECD', '<i4'), 
	('UNITCD', '<i4'), ('COUNTYCD', '<i4'), ('PLOT', '<i4'), ('PLOT_STATUS_CD', '<i4'), ('PLOT_NONSAMPLE_REASN_CD', '<i4'), ('MEASYEAR', '<i4'), 
	('MEASMON', '<i4'), ('MEASDAY', '<i4'), ('REMPER', '<f8'), ('KINDCD', '<i4'), ('DESIGNCD', '<i4'), ('RDDISTCD', '<i4'), ('WATERCD', '<i4'), 
	('LAT', '<f8'), ('LON', '<f8'), ('ELEV', '<i4'), ('GROW_TYP_CD', '<i4'), ('MORT_TYP_CD', '<i4'), ('P2PANEL', '<i4'), ('P3PANEL', '<i4'), 
	('ECOSUBCD', '<U7'), ('CONGCD', '<i4'), ('MANUAL', '<f8'), ('SUBPANEL', '<i4'), ('KINDCD_NC', '<i4'), ('QA_STATUS', '<i4'), 
	('CREATED_BY', '<U2'), ('CREATED_DATE', '<U8'), ('CREATED_IN_INSTANCE', '<U6'), ('MODIFIED_BY', '<U2'), ('MODIFIED_DATE', '<U8'), 
	('MODIFIED_IN_INSTANCE', '<U6'),  ('MICROPLOT_LOC', '<U6'), ('DECLINATION', '<f8'), ('EMAP_HEX', '<i4'), ('SAMP_METHOD_CD', '<i4'), 
	('SUBP_EXAMINE_CD', '<i4'),  ('MACRO_BREAKPOINT_DIA', '<i4'), ('INTENSITY', '<U2'), ('CYCLE', '<i4'), ('SUBCYCLE', '<i4'), 
	('ECO_UNIT_PNW', '<U2'),  ('TOPO_POSITION_PNW', 'S3'), ('NF_SAMPLING_STATUS_CD', '<i4'), ('NF_PLOT_STATUS_CD', '<i4'), 
	('NF_PLOT_NONSAMPLE_REASN_CD', '<i4'),  ('P2VEG_SAMPLING_STATUS_CD', '<i4'), ('P2VEG_SAMPLING_LEVEL_DETAIL_CD', '<i4'), 
	('INVASIVE_SAMPLING_STATUS_CD', '<i4'),  ('INVASIVE_SPECIMEN_RULE_CD', '<i4'), ('DESIGNCD_P2A', '?')])
'''
plot = np.genfromtxt(plotfile, delimiter = ',', names = True, dtype = None)

#===================================
xmax = -108
xmin = -112
ymin = 40
ymax = 46
AOA = [xmin, xmax, ymin, ymax] #specify the bounds for the FIA data
#===================================

f_plot = data_AOA(AOA, plot)
#extract the CN's
fcn = f_plot['CN']
cn = get_CN(fcn)
s_CN = get_CN(seed['PLT_CN'])

ind = find_CN_match(s_CN, cn)