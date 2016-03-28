#WLIS reanalysis....
#Author: Tony Chang

import numpy as np
from matplotlib import pyplot as plt

workspace = "E:\\WBP_model\\New_Analysis\\"
fname = "WLIS_Fullset_tab_mod_regen_wosource.csv"
filename = "%s%s"%(workspace,fname)
wlis = np.genfromtxt(filename, delimiter=',', dtype="i1, i4, U10,f4, f4, U2, i3" , names = True)

#This may not be the most useful dataset as it only describes areas where regeneration occurs, but 
#not independently of old growth. For this reason, the region where regen occurs is not describing
#a unique climate space different from that of adults....

#now check the FIA seedling database...

def data_AOA(AOA, plot):
	#returns the data filtered by bounding box AOA
	lat = plot['LAT']
	lon = plot['LON']
	lat_cut = (lat>=AOA[1]) & (lat<=AOA[3])
	lon_cut = (lon>=AOA[0]) & (lon<=AOA[2])
	return(plot[lat_cut & lon_cut])

def remove_dup(data):
	#sorts the unique CN values
	u, i = np.unique(data['CN'], return_index=True)
	return(data[i])

workspace = "E:\\FIA\\"
fname ="SEEDLING.CSV"
seedfilename = "%s%s" %(workspace,fname)
seed = np.genfromtxt(seedfilename, delimiter=',', dtype =None, names =True)
#filter out whitebark pine: 101
wbpseed = seed[np.where(seed['SPCD'] ==101)]
#remove duplicates
u, i = np.unique(wbpseed['PLT_CN'], return_index=True)
wbpseed_nodup = wbpseed[i]
scn_wbp = wbpseed_nodup['PLT_CN']
#repeat for limberpine
lbpseed = seed[np.where(seed['SPCD'] ==113)]
u, i = np.unique(lbpseed['PLT_CN'], return_index=True)
lbpseed_nodup = lbpseed[i]
scn_lbp = lbpseed_nodup['PLT_CN']

s = []
for i in wbp_scn:
	s.append(int(i.decode("utf-8")[1:-1]))
s = np.array(s)

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
ffia_nodup = remove_dup(ffia)

f = []
for i in ffia_nodup['CN']:
	f.append(int(i))
f = np.array(f)

#find the matching CNs....
index = []
for i in wbps:
	temp = np.where(f==i)[0]
	if len(temp) != 0:
		index.append(temp[0])
index = np.unique(np.array(index))

seedling_wbp = ffia_nodup[index]