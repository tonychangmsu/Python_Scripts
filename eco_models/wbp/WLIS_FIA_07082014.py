#WLIS reanalysis....
#Author: Tony Chang

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
workspace = "E:\\WBP_model\\New_Analysis\\"
fname = "WLIS_Fullset_tab_mod_regen_wosource.csv"
filename = "%s%s"%(workspace,fname)
wlis = np.genfromtxt(filename, delimiter=',', dtype="i1, i4, U10,f4, f4, U2, i3" , names = True)

#This may not be the most useful dataset as it only describes areas where regeneration occurs, but 
#not independently of old growth. For this reason, the region where regen occurs is not describing
#a unique climate space different from that of adults....
#
#some notes on FIA data as a representative sample. The standard plot consists of four 24 ft (0.0415 acres) radius subplots. So 
#in their sum may represent 0.166 acres. However climate data exists at a 800*800m^2 area, or roughly 158.147 acres 
#seedlings are measured within a 6.8 ft radius or 1/300 acre area microplots (145.267 ft^2), totaling 1/75 of an acre. 

#now check the FIA seedling database...

def data_AOA(AOA, plot):
	#returns the data filtered by bounding box AOA
	lat = plot['LAT']
	lon = plot['LON']
	lat_cut = (lat>=AOA[1]) & (lat<=AOA[3])
	lon_cut = (lon>=AOA[0]) & (lon<=AOA[2])
	return(plot[lat_cut & lon_cut])

def remove_dup(data,field):
	#returns the data without duplicates by specified field
	field_data = data[field]
	setdata = set(field_data)
	unique_data_indices = []
	for i in setdata:
		unique_data_indices.append(np.where(field_data==i)[0][0])
	unique_data_indices = np.array(unique_data_indices)
	return(data[unique_data_indices])

def decode(data, field):
	#returns the data with a field from 'bytes' to 'unicode'
	decoded = []
	for i in data[field]:
		decoded.append(int(i.decode('utf-8')[1:-1]))
	return(np.array(decoded))
	
workspace = "E:\\FIA\\"
fname ="SEEDLING.CSV"
seedfilename = "%s%s" %(workspace,fname)
seed = np.genfromtxt(seedfilename, delimiter=',', dtype =None, names =True)
#filter out whitebark pine: 101
whitebark_code = 101
wbpseed = seed[np.where(seed['SPCD'] == whitebark_code)]
unique_wbpseed = remove_dup(wbpseed, 'PLT_CN')
uni_wbpcn = decode(unique_wbpseed,'PLT_CN')

#try for limber pine: 113
limber_code = 113
lbpseed = seed[np.where(seed['SPCD'] == limber_code)]
unique_lbpseed = remove_dup(lbpseed, 'PLT_CN')
uni_lbpcn = decode(unique_lbpseed,'PLT_CN')

#==============================================================
#==============================================================
#==============================================================

indices = []
indall = []
for i in uni_cns:
	indices.append(np.where(uni_wbpcn==i)[0][0])
	indall.append(np.where(uni_wbpcn==i)[0])
indices = np.array(indices) #notes the first incident of the PLT_CN
indall = np.array(indall)

'''
#get the sapling count per area?
#microplot dimensions are 6.8 ft radius
#=======================================adding the subplot totals======================
area = np.pi * 6.8**2 #in ft^2
wbp_sap_den = []
for i in range(len(indall)):
	plot_area = len(indall[i])*area
	wbp_sap_den.append(np.sum(wbpseed['TREECOUNT_CALC'][indall[i]])/plot_area)
wbp_sap_den = np.array(wbp_sap_den)
#convert to m^2
c = 0.092903
wbp_sap_den_m2 = wbp_sap_den/c
'''

#===================Link with FIA data =============================
workspace = 'E:\\FIA\\'
name = 'fia_obs_clean4.10.2014.csv'
fiafilename = '%s%s' %(workspace,name)

fia = np.genfromtxt(fiafilename, delimiter = ',', names = True)

xmax = -108.263
xmin = -112.436
ymin = 42.252
ymax = 46.182
AOA = [xmin, ymin, xmax, ymax] #specify the bounds for the FIA data

ffia = data_AOA(AOA, fia)
ffia_wbp = remove_dup(ffia[np.where(ffia['SPCD'] == 101)],'CN')

ffia_uni = remove_dup(ffia,'CN')

no_dup_cn = ffia_uni['CN']

#find the matching CNs....
matched_index = []
for i in uni_wbpcn:
	temp = np.where(no_dup_cn==i)[0]
	if len(temp) != 0:
		matched_index.append(temp[0])
matched_index = np.array(matched_index)
wbpseedling_fia = ffia_uni[matched_index]

#repeat for limber pine
AOA2 = [-112,38,-104,46]
ffia2 = data_AOA(AOA2, fia)
fia_uni = remove_dup(ffia2,'CN')
no_dup_cn2 = fia_uni['CN']

matched_index_lbp = []
for i in uni_lbpcn:
	temp = np.where(no_dup_cn2==i)[0]
	if len(temp) != 0:
		matched_index_lbp.append(temp[0])
matched_index_lbp = np.array(matched_index_lbp)
lbpseedling_fia = fia_uni[matched_index_lbp]

#plot the seedlings
plt.scatter(ffia_wbp['LON'], ffia_wbp['LAT'], color = 'red')
plt.scatter(wbpseedling_fia['LON'], wbpseedling_fia['LAT'], color = 'blue')
plt.scatter(lbpseedling_fia['LON'], lbpseedling_fia['LAT'], color = 'yellow')

plt.hist(ffia_wbp['ELEV'], bins = 20, color = 'green', alpha = 0.5)
plt.hist(wbpseedling_fia['ELEV'], bins = 20, color = 'blue', alpha = 0.5)
plt.hist(lbpseedling_fia['ELEV'], bins = 20, color = 'yellow', alpha = 0.5)

#we might calculate the density of seedlings by area (normalize) and then perform a logistic regression on all FIA sites



