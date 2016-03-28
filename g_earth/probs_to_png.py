# convert probability csv to png 
# abstract creates a series of png files that can be added to Google Earth KMZ file to 
# generate a time series of change for WBP in the GYE.
# author: Tony Chang
# date: 9.10.14

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import gdal
import osr
from matplotlib import colors

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
models = ['CESM1-BGC']
rcps = [85]
syear = 2010
eyear = 2100
n = eyear-syear
nrows, ncols = 471,504 #dimensions of the grids for GYE

yprobs = []

for model in models:
	for rcp in range(len(rcps)): #next parameter is the rcp
		filename = workspace + model + '\\' + model + '_' + str(rcps[rcp]) + '_' + str(syear) + '-' + str(eyear-1) + '_probs.csv'
		data = np.genfromtxt(filename, delimiter = ',', names = True)
		for i in range(n):
			probs =  np.reshape(data[str(syear+i)],(nrows, ncols))
			yprobs.append(PGridData(i+syear, 'prob_pres', model, rcps[rcp], probs))

workspace = "E:\\earth\\WBP_projections\\PNG5\\"

#figy,figx = 1024,956
my_dpi = 100

cmap = colors.ListedColormap(['red', 'blue']) #two color selection
th = 0.421 #threshold for binary classification
bounds=[0,th,1]
norm = colors.BoundaryNorm(bounds, cmap.N)

cmap = get_cmap('RdBu')
for y in range(eyear-syear):
	name = "%sWBP_prob_%s_%i.png" %(workspace,models[0],y+syear)
	fig = plt.figure(frameon=False, figsize=(ncols/my_dpi, nrows/my_dpi), dpi=my_dpi)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(yprobs[y].data, cmap = cmap)#, aspect='normal')
	#ax.imshow(yprobs[y].data, cmap = cmap, norm=norm)
	plt.savefig(name, dpi = my_dpi)

alpha = 70
for y in range(eyear-syear):
	name = "%sWBP_prob_%s_%i.png" %(workspace,models[0],y+syear)
	im = Image.open(name)
	im.putalpha(alpha)
	im.save(name)
