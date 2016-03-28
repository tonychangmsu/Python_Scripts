# patch analysis of WBP modeling
# author: Tony Chang
# date: 3.25.14

import patch_tools as pt
import numpy as np
from matplotlib import pyplot as plt

class PGridData(object): #initialize function to construct probability class	
	def __init__(self, year=None, var = None, model = None, rcp = None, data=None):
		self.year = year
		self.var = var
		self.model = model
		self.rcp = rcp
		self.data = data
	def mean(self): #method to get the mean of the domain
		return(np.mean(self.data))

		
# Get the Data
#===================================================================================================================
#===================================================================================================================
workspace = "E:\\wbp_model\\prob_output_04172014\\"
models = ['HadGEM2-ES', 'HadGEM2-CC', 'HadGEM2-AO', 'CNRM-CM5', 'CMCC-CM', 'CESM1-CAM5', 'CESM1-BGC', 'CCSM4', 'CanESM2']
rcps = [45,85]
syear = 2010
eyear = 2099
n = eyear-syear+1 #number of years of data
nrows, ncols = 471,504 #dimensions of the grids for GYE

yprobs = []

for model in models:
	yprobs_model =[[],[]]
	for rcp in range(len(rcps)):
		filename = workspace + model + '\\' + model + '_' + str(rcps[rcp]) + '_' + str(syear) + '-' + str(eyear) + '_probs.csv'
		data = np.genfromtxt(filename, delimiter = ',', names = True)
		for i in range(n):
			probs =  np.reshape(data[str(syear+i)],(nrows, ncols))
			yprobs_model[rcp].append(PGridData(i+syear, 'prob_pres', model, rcps[rcp], probs))
	yprobs.append(yprobs_model)

# generate the ensemble averages and output as a tiff
	
# Determine the patch sizes

patches = []
for model in range(len(models)):
	patchdata =[[],[]]
	for rcp in range(len(rcps)):
		for i in range(n):
			patchdata[rcp].append(pt.get_patch(np.where(yprobs[model][rcp][i].data>0.421, 1,0), method = 8))
	patches.append(patchdata)

# calculate a time series for the patch number and mean

patch_stat = []
for model in range(len(models)):
	patch_model =[[],[]]
	for rcp in range(len(rcps)):
		for i in range(n):
			patch_model[rcp].append(np.array([patches[model][rcp][i][1],np.mean(patches[model][rcp][i][2]), np.std(patches[model][rcp][i][2])]))
	patch_stat.append(patch_model)
patch_stat = np.array(patch_stat)

#plot stuff number of patches
t = np.arange(syear, eyear+1)
ens_n45 = np.mean(patch_stat[:,0,:,0], axis=0)
ens_mu45 = np.median(patch_stat[:,0,:,1], axis=0)
ens_sd45 = np.mean(patch_stat[:,0,:,2], axis=0)

#repeat for rcp 85
ens_n85 = np.mean(patch_stat[:,1,:,0], axis=0)
ens_mu85 = np.median(patch_stat[:,1,:,1], axis=0)
ens_sd85 = np.mean(patch_stat[:,1,:,2], axis=0)

'''
for i in range(len(models)):
	plt.plot(t,patch_stat[i,0,:,0])
plt.plot(t,ens_n45, lw = 3, color = 'red') #plot of n patches

# mean patch size
for i in range(len(models)):
	plt.plot(t,patch_stat[i,0,:,1])
plt.plot(t,ens_mu45, lw = 3, color = 'red') #plot of n patches
'''
#create a range for 2040-2099
s = 0 #note if you want the 2010 to 2099 range, use s = 0
ax1 = plt.subplot(211)
plt.grid()
ax1.plot(t[s:], ens_n45[s:], color = 'blue', label = 'RCP 4.5 no. patches', lw= 2)
ax1.set_ylabel('Number of patches', color = 'blue')
for tl in ax1.get_yticklabels():
	tl.set_color('b')
ax1.set_xlabel('Year')
ax2 = ax1.twinx()
ax2.plot(t[s:], (ens_mu45*(.8**2))[s:], color = 'red', ls ='--', label = 'RCP 4.5 Median patch size', lw= 2)
ax2.set_ylabel('Median patch size $(km^2)$', color = 'red')
for tl in ax2.get_yticklabels():
	tl.set_color('r')
plt.title('GCM ENS_AVG RCP 4.5 WBP patch projections')

ax12 = plt.subplot(212)
plt.grid()
ax12.plot(t[s:], ens_n85[s:],  color = 'blue', label = 'RCP 8.5 no. patches', lw= 2)
ax12.set_ylabel('Number of patches', color = 'blue')
for tl in ax12.get_yticklabels():
	tl.set_color('b')
ax12.set_xlabel('Year')
ax22 = ax12.twinx()
ax22.plot(t[s:], (ens_mu85*(.8**2))[s:], color = 'red', ls ='--', label = 'RCP 8.5 Median patch size', lw= 2)
ax22.set_ylabel('Median patch size $(km^2)$', color = 'red')
for tl in ax22.get_yticklabels():
	tl.set_color('r')
plt.title('GCM ENS_AVG RCP 8.5 WBP patch projections')
plt.subplots_adjust(hspace=0.4)
plt.savefig('ENS_AVG_patches_04182014_a.png', bbox_inches='tight')
plt.show()
#===================
#3D plot?
'''
from mpl_toolkits.mplot3d import Axes3D
fig= plt.figure()
ax = fig.add_subplot(111, projection='3d')

xpos = t
ypos = ens_mu
zpos = np.zeros(n)
dx = np.ones(n)
dy = np.ones(n)
dz = ens_n
ax.bar3d(xpos,ypos,zpos,dx,dy,dz, color = 'orange', alpha = 0.9)
for i in range(len(models)):
	dx = np.ones(n)
	dy = np.ones(n)
	ypos = patch_stat[i,0,:,1]
	dz = patch_stat[i,0,:,0]
	ax.bar3d(xpos,ypos,zpos,dx,dy,dz, color = 'blue', alpha = 0.4,edgecolor = 'none')
'''
	
##############################################################
#Additional post analysis 

path = 'E:\WBP_model\output\prob\wbp_prob_2010.tif'
ds = gdal.Open(path)
yprobs_hi = np.array(ds.GetRasterBand(1).ReadAsArray()) # this is the 2010 probabilities
thresh = 0.421
baseline = len(np.where(yprobs_hi >thresh)[0])

#gather all the prob data
threshcells45 = []
#looking for the variance terms
for i in range(n+1):
	yearprobs45 = []
	for m in range(len(models)):
		yearprobs45.append(len(np.where(yprobs[m][0][i].data>thresh)[0])/baseline*100)
	threshcells45.append(yearprobs45)
threshcells45 = np.array(threshcells45)

threshcells85 = []
#looking for the variance terms
for i in range(n+1):
	yearprobs85 = []
	for m in range(len(models)):
		yearprobs85.append(len(np.where(yprobs[m][1][i].data>thresh)[0])/baseline*100)
	threshcells85.append(yearprobs85)
threshcells85 = np.array(threshcells85)

t45sd = np.std(threshcells45, axis=1)
t85sd = np.std(threshcells85, axis=1)

t = np.arange(syear, eyear+1)
#plt.plot(t, t45sd, lw = 2, color = 'orange', label='RCP 4.5')
#plt.plot(t, t85sd, lw = 2, color = 'red', ls = '--', label='RCP 8.5')
plt.scatter(t, t45sd, color = 'orange', label='RCP 4.5')
plt.scatter(t, t85sd, color = 'red', label='RCP 8.5')
plt.grid()
plt.xlabel('Year')
plt.ylabel('$\sigma$ (percent area) ')
plt.legend()
plt.savefig('sd_plot.png', bbox_inches='tight')