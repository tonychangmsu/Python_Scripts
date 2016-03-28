import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import gdal as gdal
import datetime as datetime
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import os
from datetime import date

# Author: Tony Chang
# Abstract: Random forest model for seedling WBP within GYE
# Date: 10/15/2014

class PGridData(object): #initialize function to construct probability class	
	def __init__(self, year=None, var = None, model = None, rcp = None, data=None):
		self.year = year
		self.var = var
		self.model = model
		self.rcp = rcp
		self.data = data
	def mean(self): #method to get the mean of the domain
		return(np.mean(self.data))

############################################################################################
################################		MODELING 			################################
############################################################################################

def rfFit(sampledata, mf = None):
# performs RF analysis and returns fitted model
	y_rp = sampledata['response']
	x_rp = sampledata.drop('response', 1)
	#specify the random forest parameters
	clasf = RandomForestClassifier(n_estimators=1000, criterion = 'gini',compute_importances = True, max_features= mf, min_samples_leaf=20,	oob_score=True, bootstrap = True)
	rfmodel = clasf.fit(x_rp, y_rp)
	return(rfmodel)

def rfAnalysis(rfmodel, testdata, nrow, ncol):
	xt_rp = testdata
	ytestprob = rfmodel.predict_proba(xt_rp)
	ytestprob_0 = ytestprob[:,0].reshape((nrow,ncol))
	ytestprob_1 = ytestprob[:,1].reshape((nrow,ncol))
	return([ytestprob_0, ytestprob_1])

def rfTests(sampledata, rfmodel):
# performs RF analysis and returns an array
# element 1: feature labels array and importance array
# element 2: confusion matrix value array and percentage correct
# element 3: false positive rate, true positive rate, receiver operator curve
	y_rp = sampledata['response']
	x_rp = sampledata.drop('response', 1)
	
	#feature importance
	xfname= x_rp.columns.values
	featureimp = rfmodel.feature_importances_
	sxfeature = xfname[np.argsort(featureimp)][::-1]
	sfeatureimp = np.sort(featureimp)[::-1]
	
	#confusion matrix
	ypred = rfmodel.predict(x_rp)
	cmx = confusion_matrix(y_rp, ypred)
	cmxperc = np.array([cmx[0]/np.sum(cmx[0]), cmx[1]/np.sum(cmx[1])])

	#roc curve
	ypred_prob = rfmodel.predict_proba(x_rp)
	fpr, tpr, thresholds = roc_curve(y_rp, ypred_prob[:,1])
	roc_auc = sklearn.metrics.auc(fpr,tpr)
	return([sxfeature,sfeatureimp],[cmx,cmxperc],[fpr,tpr,thresholds, roc_auc])

############################################################################################
#######################			plotting functions 		####################################
############################################################################################

def rfImportancePlot(sxfeature,sfeatureimp,savefig='n'):
	#====plot routines====
	#feature importance plot
	#plt.subplot2grid((3,3),(0,2), rowspan=2)
	pos = np.arange(len(sxfeature))+0.5
	plt.barh(pos, sfeatureimp[::-1], align='center')
	plt.yticks(pos, sxfeature[::-1])
	plt.xlabel('Mean decrease in accuracy')
	plt.ylabel('Feature')
	plt.title('Feature importance plot')
	plt.grid()
	if savefig!='n':
		date = datetime.datetime.now().strftime('%Y%m%d')
		plt.savefig('importance_%s.png' %(date), bbox_inches='tight') 
	return()

def rfROCplot(fpr,tpr,roc_auc,savefig='n'):
	#roc curve plot
	#plt.subplot2grid((3,3),(2,2))
	plt.plot(fpr,tpr,label = 'ROC curve (area = %0.2f)' %roc_auc)
	plt.plot([0,1],[0,1], 'k--')
	plt.xlim([-0.01,1.01])
	plt.ylim([-0.01,1.01])
	plt.xlabel('False Positive Rate', fontsize=14)
	plt.ylabel('True Positive Rate', fontsize=14)
	plt.title('Receiver operating characteristic plot')
	plt.grid()
	plt.legend(loc='lower right')
	if savefig!='n':
		date = datetime.datetime.now().strftime('%Y%m%d')
		plt.savefig('ROC_%s.png' %(date), bbox_inches ='tight')
	return()

############################################################################################
###########################		Accuracy assessment		####################################
############################################################################################

def kappa(po,pe):
	return((po-pe)/(1-pe))

def po(p, sn, sp):
	return(p*sn+((1-p)*sp))

def pe(p, po, sn, sp):
	return(-2 * (sn + sp - 1) * p * (1 - p) + po)

def prevalence(sampledata):
	#calculates the prevelance
	y_rp = sampledata['response']
	pr = len(np.where(y_rp==1)[0])
	return(pr/len(y_rp))
	
def plotAccuracy(fsampledata,rfmodel,savefig='n'):
	[sxfeature,sfeatureimp],[cmx,cmxperc],[fpr,tpr,thresholds,roc_auc] = rfTests(fsampledata,rfmodel) #test the rfmodel
	sn = tpr
	sp = (fpr-1)/-1
	d = np.abs(sn-sp)
	thr = np.where(d==np.min(d))
	tss = sp+sn-1
	tss_i = np.where(tss==np.max(tss))
	p = prevalence(fsampledata)
	p_o = po(p, sn, sp)
	p_e = pe(p, p_o, sn, sp)
	k = kappa(p_o, p_e)
	k_i = np.where(k==np.max(k))
	plt.plot(thresholds,sn, lw = 2, label = 'Sensitivity $(s_n)$')
	plt.plot(thresholds,sp, lw = 2, label = 'Specificity $(s_p)$')
	plt.vlines(thresholds[thr],0,1, color = 'red', linewidth = 1.5, label = '$s_n/s_p = 1:$ $%0.3f$' %(thresholds[thr]))
	plt.vlines(thresholds[tss_i],0,1, color = 'orange', linewidth = 1.5, linestyle = '--', label = 'Max TSS: $%0.3f$' %thresholds[tss_i])
	plt.vlines(thresholds[k_i],0,1, color = 'purple', linewidth = 1.5, linestyle = '-.', label = 'Max kappa: $%0.3f$' %thresholds[k_i])
	leg = plt.legend(fancybox=True,fontsize=12)
	leg.get_frame().set_alpha(0.5)
	plt.xlabel('Probability of Presence')
	plt.grid()
	if savefig !='n':
		date = datetime.datetime.now().strftime('%Y%m%d')
		plt.savefig('E:\\WBP_model\\New_Analysis\\out\\accuracy_%s.png' %(date), bbox_inches ='tight')
	return(thresholds[thr])

############################################################################################
#######################			Build projections			################################
############################################################################################

def projModel(m, r, year, flist, rfmodel):
	models = ['HadGEM2-ES', 'HadGEM2-CC', 'HadGEM2-AO', 'CNRM-CM5', 'CMCC-CM', 'CESM1-CAM5', 'CESM1-BGC', 'CCSM4', 'CanESM2']
	rcps = [45,85]
	pfilename = 'E:\\WBP_model\\projections_04172014\\%s\\%s_%s_%s_%s_data.csv' %(models[m],models[m],str(rcps[r]),str(year-30),str(year))
	projectiondata = pd.read_csv(pfilename)
	fprojectiondata = projectiondata[flist[1:]]
	yp = rf_analysis(rfmodel, fprojectiondata)
	pr = PGridData(year, 'prob_pres', models[m], rcps[r], yp[1])
	return(pr)

############################################################################################
###########################		MODEL MAIN		############################################
############################################################################################

working_directory = 'e:\\wbp_model\\New_Analysis'
os.chdir(working_directory) #change directories to get everything in the same place

#get the data
sampledata = pd.read_csv('E:\\WBP_model\\New_Analysis\\FIA_merged_cleaned.csv')
flist = ['tmax7', 'pack4', 'vpd3', 'ppt9', 'tmin1', 'aet7', 'ppt4', 'pet8'] 
testdata = pd.read_csv('E:\\WBP_model\\fielddata\\PRISM_1980_2010_data.csv')

def runModel(sampledata, testdata, flist):
	fsampledata = sampledata[['response']+flist]
	ftestdata = testdata[flist]
	rfmodel = rfFit(fsampledata)
	dem = gdal.Open('E:\\GYE_TOPO\\dem_800m.tif') #get the ncol and nrow size from the dem data
	ncol = dem.RasterXSize
	nrow = dem.RasterYSize
	probs = rfAnalysis(rfmodel, ftestdata, nrow, ncol)
	return(probs, rfmodel, fsampledata, ftestdata) #output the probability of presence surface for 2010

probs, rfmodel, fsampledata, ftestdata = runModel(sampledata, testdata, flist)

#save the probability surface
size = np.shape(probs[1])[0]*np.shape(probs[1])[1]
p_2010 = np.reshape(probs[1], (size))
lat = testdata['LAT']
lon = testdata['LON']
outprob = pd.DataFrame(np.array([p_2010,lat,lon]).T, columns = ['p_seed_2010','lat','lon'])
dt = date.today().strftime("%m%d%y")
#outprob.to_csv('E:\\WBP_model\\New_Analysis\\out\\seed_probs_2010_%s.csv'%dt)

#open the adult surface and plot next to each other
adult = pd.read_csv('E:\\WBP_model\\output\\PRISM_2010_probs.csv')
adult_prob = np.reshape(adult['2010'], np.shape(probs[1]))
seed_prob = probs[1]
xmin = -112.39583333837999 #112 23 45
xmax = -108.19583334006 #108 11 45
ymin = 42.279166659379996 #42 16 45
ymax = 46.195833324479999 #46 11 45
ae = [xmin, xmax,ymin, ymax]
plt.rcParams['figure.figsize'] = 16,12
plt.subplot(121)
plt.imshow(adult_prob, vmin = 0, vmax = .7, extent = ae)
plt.title('P.albicaulis adult 2010') 
plt.xlabel('Longitude (dd)')
plt.ylabel('Latitude (dd)')
plt.subplot(122)
plt.imshow(seed_prob, vmin = 0, vmax = .7, extent = ae)
plt.title('P.albicaulis seedling 2010')
plt.xlabel('Longitude (dd)')
plt.ylabel('Latitude (dd)')
plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)

cax = plt.axes([0.12,0.25, 0.68, 0.02]) #axes represent the left, bottom, width, and height 
cbar = plt.colorbar(cax=cax, orientation = 'horizontal')
cbar.set_label('Percent bioclimate suitability', size = 16)

#plt.savefig('E:\\WBP_model\\New_Analysis\\out\\probs%s.png'%dt, bbox_inches='tight')

#generate binary plots with the threshold values
adult_bin = np.where(adult_prob <=0.421, 0, 1)
seed_bin = np.where(seed_prob <= 0.298, 0, 1)
plt.subplot(121)
plt.imshow(adult_bin, vmin = 0, vmax = 1, extent = ae)
plt.title('P.albicaulis adult 2010') 
plt.xlabel('Longitude (dd)')
plt.ylabel('Latitude (dd)')
plt.subplot(122)
plt.imshow(seed_bin, vmin = 0, vmax = 1, extent = ae)
plt.title('P.albicaulis seedling 2010')
plt.xlabel('Longitude (dd)')
plt.ylabel('Latitude (dd)')
plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)

cax = plt.axes([0.12,0.25, 0.68, 0.02]) #axes represent the left, bottom, width, and height 
cbar = plt.colorbar(cax=cax, orientation = 'horizontal')
cbar.set_label('Percent bioclimate suitability', size = 16)
plt.savefig('E:\\WBP_model\\New_Analysis\\out\\binary%s.png'%dt, bbox_inches='tight')

#do the analysis for the 3 climate periods again.
def getProjection(m,r,year):
	#input model are determined by the index of the models and r is the index of the rcp list
	models = ['HadGEM2-ES', 'HadGEM2-CC', 'HadGEM2-AO', 'CNRM-CM5', 'CMCC-CM', 'CESM1-CAM5', 'CESM1-BGC', 'CCSM4', 'CanESM2']
	rcps = [45,85]
	pfilename = 'E:\\WBP_model\\projections_04172014\\%s\\%s_%s_%s_%s_data.csv' %(models[m],models[m],str(rcps[r]),str(year-30),str(year))
	projectiondata =  pd.read_csv(pfilename)
	return(projectiondata)

#get the rcp 45 outputs for 3 climate periods, 1:2040, 2:2070, 3:2099
years = [2040, 2070, 2099]
m = 6 # representing CESM1-BGC
pr_probs = [[],[]]
for r in range(2): #both rcps
	for y in years:
		pdata = getProjection(m, r, y)
		pr, rfm, fs, ft = runModel(sampledata, pdata, flist)
		pr_probs[r].append(pr[1]) #we are only interested in the probability of presence counts
		
fig = plt.figure()
rcp =['4.5','8.5']
c = 1
for i in range(2):
	for j in range(len(pr_probs[i])):
		ax = plt.subplot(2,3,c)
		out = ax.imshow(pr_probs[i][j], vmin = 0, vmax = .6, extent = ae)
		ax.set_xticks(np.arange(-112,-108))
		plt.xlabel('Longitude (dd)')
		ax.yaxis.tick_right()
		if j ==0:
			ax.set_ylabel('RCP %s'%rcp[i], fontsize = 24, fontweight = 'bold')
		else:
			ax.yaxis.set_label_position("right")
			ax.set_ylabel('Latitude (dd)')
		plt.title(years[j], fontsize = 24, fontweight = 'bold')
		c+=1
cax = fig.add_axes([0.17,0.08, 0.68, 0.02])
cbar = fig.colorbar(out, cax=cax, orientation = 'horizontal')
cbar.set_label('Percent bioclimate suitability (CESM1-BGC)', size = 16)
plt.savefig('E:\\WBP_model\\New_Analysis\\out\\projection_probs%s.png'%dt, bbox_inches='tight')
'''
full_sample = sampledata.drop(['Unnamed: 0', 'Unnamed: 0.1', 'LAT', 'LON', 'ELEV', 'dem'], 1)
full_test = testdata.drop(['Unnamed: 0','LAT', 'LON'],1)
full_rfmodel = rfFit(full_sample)
#get the number of rows and cols from the dem data
dem = gdal.Open('E:\\GYE_TOPO\\dem_800m.tif')
ncol = dem.RasterXSize
nrow = dem.RasterYSize
full_probs = rfAnalysis(full_rfmodel, full_test, nrow, ncol)
'''
#plots from the full list
full_rfmodel_tr = rfTests(full_sample, full_rfmodel)
rfImportancePlot(full_rfmodel_tr[0][0],full_rfmodel_tr[0][1])
rfROCplot(full_rfmodel_tr[2][0],full_rfmodel_tr[2][1],full_rfmodel_tr[2][3])

########################################################################################
############################Consider a filtered list####################################
########################################################################################

thr, prob, rfmodel, fsample,ftestdata = runModel(sampledata, testdata, flist)

########################################################################################
############################		POST - PROCESSING		############################
########################################################################################
models = ['HadGEM2-ES', 'HadGEM2-CC', 'HadGEM2-AO', 'CNRM-CM5', 'CMCC-CM', 'CESM1-CAM5', 'CESM1-BGC', 'CCSM4', 'CanESM2']
rcps = [45,85]
yprobs = []
syear = 2010
eyear = 2100
counter = 1
ptotal = len(models)*len(rcps)*(eyear-syear)
draw_progress_bar(counter/ptotal)
timebegin = time.clock()
for m in range(len(models)):
	yprobs_model = [[],[]]
	for r in range(len(rcps)):
		for year in range(syear, eyear):
			pr = proj_model(m,r,year,flist, rfmodel)
			yprobs_model[r].append(pr)	
			counter += 1
			draw_progress_bar(counter/ptotal)
	yprobs.append(yprobs_model)		
timeend = time.clock()
draw_progress_bar(1)
print('\n')
print('process complete: ' + str(timeend-timebegin)+ ' seconds')