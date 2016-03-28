import time
import sys
import numpy as np
import sklearn
import gdal
from gdalconst import *
import osr 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

class PGridData(object): #initialize function to construct probability class	
	def __init__(self, year=None, var = None, model = None, rcp = None, data=None):
		self.year = year
		self.var = var
		self.model = model
		self.rcp = rcp
		self.data = data
	def mean(self): #method to get the mean of the domain
		return(np.mean(self.data))

def header_extract():
	#takes arbitrary PRISM dataset and extracts the header parameters
	filename = "E:\\PRISM\\tmin\\PRISM800m_tmin1895_1.tif" 
	dataset = gdal.Open(filename, GA_ReadOnly)
	ncols = dataset.RasterXSize
	nrows = dataset.RasterYSize
	bands = dataset.RasterCount
	driver = dataset.GetDriver().LongName
	geotransform = dataset.GetGeoTransform()
	xul = geotransform[0]
	yul = geotransform[3]
	csize = geotransform[1]
	header = {'ncols':ncols, 'nrows':nrows,'bands':bands,'driver':driver, 'xul':xul, 'yul':yul, 'csize':csize}
	return(header) #returns header as directory for readability

def topo_extract():
    aspectPath = "E:\\GYE_TOPO\\aspect_800m.tif"
    slopePath = "E:\\GYE_TOPO\\slope_800m.tif"
    elevPath = "E:\\GYE_TOPO\\dem_800m.tif"   
    ds = gdal.Open(aspectPath)
    aspect = np.array(ds.GetRasterBand(1).ReadAsArray())
    #aspect = ds.GetRasterBand(1).ReadAsArray()
    ds = gdal.Open(slopePath)
    slope = np.array(ds.GetRasterBand(1).ReadAsArray())
    #slope = ds.GetRasterBand(1).ReadAsArray()
    ds = gdal.Open(elevPath)
    elev = np.array(ds.GetRasterBand(1).ReadAsArray())
    #elev = ds.GetRasterBand(1).ReadAsArray()
    ds = None #close files
    return(aspect,slope,elev)    

def draw_progress_bar(percent, barLen=20): #a function to display progress 
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()
	
#===============================================================================#
#===========================analysis functions==================================#
#===============================================================================#
def covariate_filter(data, filterlist, response='n'):
	#if there is a response field, then adding response ='y' will add the field
	f_dtype = []
	if (response == 'y'):
		f_dtype.append(('response','f8'))	
	f_dtype.append(('lat','f8'))
	f_dtype.append(('lon','f8')) #attach the lat and lon coordinates in the filtered dataset
	for i in range(len(filterlist)):
		f_dtype.append((filterlist[i],'f8'))
	f_data = np.zeros(len(data), dtype = f_dtype)
	for i in range(len(f_dtype)):
		f_data[f_dtype[i][0]] = data[f_dtype[i][0]]
	return(f_data)
	
def rf_fit(sampledata, mf = None):
# performs RF analysis and returns fitted model
# the sampled data should have 'response' in the first column, and lat/lon as the 2nd and 3rd columns
	if mf == None:
		mf = round(len(sampledata.dtype.names)/2) #if no maximum feature size is specified, then use half of the predictors
	y_rp = sampledata['response']
	y_rp = np.where(y_rp >0, 1, 0) #make integer?
	x_rp = []
	for i in range(3,len(sampledata.dtype.names)):
		x_rp.append(sampledata[sampledata.dtype.names[i]])
	x_rp = np.array(x_rp).T #change to np array and transpose
	clasf = RandomForestClassifier(n_estimators=1000, criterion = 'gini',compute_importances = True, max_features= mf, min_samples_leaf=20, oob_score=True, bootstrap = True)
	rfmodel = clasf.fit(x_rp, y_rp) #fit model with random forest classifier for presence case
	return(rfmodel)

def rf_analysis(rfmodel, testdata):
	#data to be analyzed by the RF model should have 'lat' and 'lon' as the 2nd and 3rd columns
	xt_rp = []
	for i in range(2,len(testdata.dtype.names)):
		xt_rp.append(testdata[testdata.dtype.names[i]])
	xt_rp = np.array(xt_rp).T
	h = header_extract()
	nrow= h['nrows']
	ncol= h['ncols']
	ytestprob = rfmodel.predict_proba(xt_rp)
	ytestprob_0 = ytestprob[:,0].reshape((nrow,ncol))
	ytestprob_1 = ytestprob[:,1].reshape((nrow,ncol))
	return([ytestprob_0, ytestprob_1])

def rf_tests(sampledata, rfmodel):
# performs RF analysis and returns an array
# element 1: feature labels array and importance array
# element 2: confusion matrix value array and percentage correct
# element 3: false positive rate, true positive rate, receiver operator curve
	y_rp = sampledata['response']
	x_rp = []
	for i in range(3,len(sampledata.dtype.names)):
		x_rp.append(sampledata[sampledata.dtype.names[i]])
	x_rp = np.array(x_rp).T #change to np array and transpose
	
	#feature importance
	xfname= np.array(sampledata.dtype.names[3:])
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

#===============================================================================#
#===========================plotting functions==================================#
#===============================================================================#
def rf_importanceplot(sxfeature,sfeatureimp):
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
	plt.savefig('importance_04202014.png', bbox_inches='tight') 
	return()

def rf_ROCplot(fpr,tpr,roc_auc,savefig='n'):
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
		plt.savefig('ROC04202014.png', bbox_inches ='tight')
	return()

def rf_plot(ytestprob, save='n', plotpoints='n'):
	#probability map plot
	#takes in probability grid class
	xmin = -112.39583333837999
	xmax = -108.19583334006
	ymin = 42.279166659379996
	ymax = 46.195833324479999
	ae = [xmin, xmax,ymin, ymax]
	lat = sampledata['lat']
	lon = sampledata['lon']
	lat1 = lat[np.where(y_rp==1)]
	lon1 = lon[np.where(y_rp==1)]
	lat0 = lat[np.where(y_rp==0)]
	lon0 = lon[np.where(y_rp==0)]
	pp = plt.imshow(ytestprob.data, extent = ae)
	cbar = plt.colorbar(pp)
	cbar.set_label('Probability of presence')
	if (plotpoints=='y'):
		p1 = plt.scatter(lon1,lat1, s=20, marker ='x', color = 'green', alpha = 0.9, label = 'field presence')
		p0 = plt.scatter(lon0,lat0, s= 7, marker = 'o', color='gray', alpha = 0.3, label = 'field absence')
		plt.legend((p1,p0), (p1.get_label(), p0.get_label()),loc ='upper right')
	plt.xlabel('Longitude (DD)')
	plt.ylabel('Latitude (DD)')
	#plt.title("RF modeled Whitebark pine species distribution map for >=8\" DBH individuals")
	plt.grid()
	if save != 'n':
		filename = ytestprob.model + '_RCP' + str(ytestprob.rcp) + '_' + str(ytestprob.year) + '04212014.png'
		plt.savefig(filename, bbox_inches = 'tight')
		print(filename + ' saved')
	#plt.subplots_adjust(wspace=0.4, hspace=0.5)
	return()

#================================================================================#
#=================================projection analysis============================#
#================================================================================#
def percent_series(yprobs, rcp, model, threshhold):
	#returns counts of cells with probabilities above specified threshhold
	for i in range(len(yprobs)):
		if(yprobs[i][0][0].model==model):
			break
	for j in range(len(yprobs[i])):
		if(yprobs[i][j][0].rcp==rcp):
			break
	output = []
	for k in range(len(yprobs[i][j])):
		count = len(yprobs[i][j][k].data[yprobs[i][j][k].data >= threshhold])
		output.append(count)
	return(np.array(output))

def plot_prob(yprob, model, rcp, year, extent):
	for i in range(len(yprobs)):
		if(yprobs[i][0][0].model==model):
			break
	for j in range(len(yprobs[i])):
		if(yprobs[i][j][0].rcp==rcp):
			break
	for k in range(len(yprobs[i][j])):
		if(yprobs[i][j][k].year==year):
			break
	selection = yprobs[i][j][k].data
	plt.imshow(selection, extent = extent)
	plt.xlabel('Longitude (dd)')
	plt.ylabel('Latitude (dd)')
	plt.grid()
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=14) 
	cbar.set_label('Probability of presence', size = 16)
	return()

#================================================================================#
#=================================MAIN===========================================#
#================================================================================#
'''
fullvarlist =  ['tmin1', 'tmin2', 'tmin3', 'tmin4', 'tmin5', 'tmin6', 'tmin7', 'tmin8', 'tmin9', 'tmin10', 'tmin11', 'tmin12',
 'tmax1', 'tmax2', 'tmax3', 'tmax4', 'tmax5', 'tmax6', 'tmax7', 'tmax8', 'tmax9', 'tmax10', 'tmax11', 'tmax12',
 'ppt1', 'ppt2', 'ppt3', 'ppt4', 'ppt5', 'ppt6', 'ppt7', 'ppt8', 'ppt9', 'ppt10', 'ppt11', 'ppt12',
 'aet1', 'aet2', 'aet3', 'aet4', 'aet5', 'aet6', 'aet7', 'aet8', 'aet9', 'aet10', 'aet11', 'aet12',
 'pet1', 'pet2', 'pet3', 'pet4', 'pet5', 'pet6', 'pet7', 'pet8', 'pet9', 'pet10', 'pet11', 'pet12',
 'pack1','pack2', 'pack3', 'pack4', 'pack5', 'pack6', 'pack7', 'pack8','pack9', 'pack10', 'pack11', 'pack12', 
 'soilm1', 'soilm2', 'soilm3', 'soilm4', 'soilm5', 'soilm6', 'soilm7', 'soilm8', 'soilm9', 'soilm10', 'soilm11', 'soilm12',
 'vpd1', 'vpd2', 'vpd3', 'vpd4', 'vpd5', 'vpd6', 'vpd7', 'vpd8', 'vpd9', 'vpd10', 'vpd11', 'vpd12']
'''
#generate the test and sample dataset
sampledata = np.genfromtxt('E:\\WBP_model\\fielddata\\1950_1980_merged_data.csv', delimiter = ',', names=True)
testdata = np.genfromtxt('E:\\WBP_model\\fielddata\\PRISM_1950_1980_data.csv', delimiter =',', names=True)
flist = ['tmax7', 'pack4', 'vpd3', 'ppt9', 'tmin1', 'aet7', 'ppt4', 'pet8'] 
'''
#subset the sample data to only inspect some covariates
#flist = ['tmax8', 'pack5', 'pack10', 'vpd6', 'tmin5', 'aet4', 'pet5', 'tmin2']
#selection with low correlations and PCA loading
#flist = np.array(['pet8', 'aet2', 'pack2', 'pet7', 'aet9', 'aet8', 'pack6', 'aet12', 'tmax4', 'aet5', 'ppt8', 'pack3', 'pet6', 'ppt10', 'tmax2', 'pet1', 'pack5', 'tmin7', 'ppt4', 'pet10', 'ppt9', 'pack8', 'tmax11', 'ppt6', 'aet7', 'ppt7', 'ppt3', 'pack7', 'tmax7', 'aet1', 'tmax3','ppt11', 'pet12', 'pet9', 'aet10', 'pet2', 'pack1', 'pet4']) #from SVD
#flist2 = ['tmax8', 'pack5', 'pack10', 'vpd6', 'tmax10', 'ppt4', 'vpd11', 'tmin4','aet4', 'pet5', 'tmin2']
#flist = ['tmin1', 'tmin2', 'tmin3', 'tmin4', 'tmin5', 'tmin6', 'tmin7', 'tmin8', 'tmin9', 'tmin10', 'tmin11', 'tmin12', 'tmax1', 'tmax2', 'tmax3', 'tmax4', 'tmax5', 'tmax6', 'tmax7', 'tmax8', 'tmax9', 'tmax10', 'tmax11', 'tmax12',  'ppt1', 'ppt2', 'ppt3', 'ppt4', 'ppt5', 'ppt6', 'ppt7', 'ppt8', 'ppt9', 'ppt10', 'ppt11', 'ppt12',  'aet1', 'aet2', 'aet3', 'aet4', 'aet5', 'aet6', 'aet7', 'aet8', 'aet9', 'aet10', 'aet11', 'aet12',  'pet1', 'pet2', 'pet3', 'pet4', 'pet5', 'pet6', 'pet7', 'pet8', 'pet9', 'pet10', 'pet11', 'pet12',  'pack1','pack2', 'pack3', 'pack4', 'pack5', 'pack6', 'pack7', 'pack8','pack9', 'pack10', 'pack11', 'pack12',   'soilm1', 'soilm2', 'soilm3', 'soilm4', 'soilm5', 'soilm6', 'soilm7', 'soilm8', 'soilm9', 'soilm10','soilm11', 'soilm12',  'vpd1', 'vpd2', 'vpd3', 'vpd4', 'vpd5', 'vpd6', 'vpd7', 'vpd8', 'vpd9', 'vpd10', 'vpd11', 'vpd12']
'''
y_rp = sampledata['response']
fsampledata = covariate_filter(sampledata, flist, 'y')
ftestdata = covariate_filter(testdata, flist)
rfmodel = rf_fit(fsampledata)
#==============================================================================#
#=========================Accuracy assessment==================================#
#==============================================================================#

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
	
def plot_accuracy(fsampledata,rfmodel):
	[sxfeature,sfeatureimp],[cmx,cmxperc],[fpr,tpr,thresholds,roc_auc] = rf_tests(fsampledata,rfmodel) #test the rfmodel
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
	plt.vlines(thresholds[thr],0,1, color = 'red', linewidth = 1.5, label = '$s_n/s_p = 1$' )
	plt.vlines(thresholds[tss_i],0,1, color = 'orange', linewidth = 1.5, linestyle = '--', label = 'Max TSS: $%0.3f$' %thresholds[tss_i])
	plt.vlines(thresholds[k_i],0,1, color = 'purple', linewidth = 1.5, linestyle = '-.', label = 'Max kappa: $%0.3f$' %thresholds[k_i])
	leg = plt.legend(fancybox=True,fontsize=12)
	leg.get_frame().set_alpha(0.5)
	plt.xlabel('Probability of Presence')
	plt.grid()
	plt.savefig('accuracy04202014.png', bbox_inches ='tight')
	return()

	
#==============================================================================#
#=========================Build projections====================================#
#==============================================================================#
def proj_model(m,r,year, flist, rfmodel):
	models = ['HadGEM2-ES', 'HadGEM2-CC', 'HadGEM2-AO', 'CNRM-CM5', 'CMCC-CM', 'CESM1-CAM5', 'CESM1-BGC', 'CCSM4', 'CanESM2']
	rcps = [45,85]
	pfilename = 'E:\\WBP_model\\projections_04172014\\%s\\%s_%s_%s_%s_data.csv' %(models[m],models[m],str(rcps[r]),str(year-30),str(year))
	projectiondata = np.genfromtxt(pfilename, delimiter =',', names = True)
	fprojectiondata = covariate_filter(projectiondata, flist)
	yp = rf_analysis(rfmodel, fprojectiondata)
	pr = PGridData(year, 'prob_pres', models[m], rcps[r], yp[1])
	return(pr)
	
yp = rf_analysis(rfmodel, ftestdata)
p = PGridData(2010, 'prob_pres', 'PRISM', None, yp[1])
yprobs_hi = [p]
#plot reference
xmin = -112.39583333837999 #112 23 45
xmax = -108.19583334006 #108 11 45
ymin = 42.279166659379996 #42 16 45
ymax = 46.195833324479999 #46 11 45
ae = [xmin, xmax,ymin, ymax]
pres = [sampledata['lon'][sampledata['response']==1],sampledata['lat'][sampledata['response']==1]]
abse = [sampledata['lon'][sampledata['response']==0],sampledata['lat'][sampledata['response']==0]]
plt.rcParams['figure.figsize'] =20,14
plt.imshow(yprobs_hi[0].data, extent = ae)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.xlabel('Longitude (dd)', size =20)
plt.ylabel('Latitude (dd)', size =20)
plt.grid()
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14) 
cbar.set_label('Probability of presence', size = 16)
plt.scatter(pres[0],pres[1], color = 'green', marker = "o" , alpha = 0.9, label ='presence')
plt.scatter(abse[0],abse[1], color = 'purple', marker = 'x',alpha = 0.9, label='absence')
plt.legend(prop={'size':22})
plt.savefig('2010WBP_plot.tif',bbox_inches = 'tight')

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

#first index element is the model
#second index is the rcp
#third index is the 30 year period

#=============================================================================#
#===========================SAVING DATA=======================================#
#=============================================================================#
#need to save this output....

workspace = "E:\\wbp_model\\prob_output_04172014\\"
def save_probs(yprobs, workspace):
	size = np.shape(yprobs[0][0][0].data)[0]*np.shape(yprobs[0][0][0].data)[1]
	yrs = len(yprobs[0][0])
	for i in range(len(yprobs)):
		model = yprobs[i][0][0].model
		for j in range(len(yprobs[i])):
			rcp = str(yprobs[i][j][0].rcp)
			lat = testdata['lat']
			lon = testdata['lon']
			labels = ['lat','lon']
			outfile = np.vstack((lat,lon))
			for k in range(len(yprobs[i][j])):
				labels.append(str(yprobs[i][j][k].year))
				pd = np.reshape(yprobs[i][j][k].data, size)
				outfile = np.vstack((outfile,pd))
			filename = workspace + model + '\\' + model + '_' + rcp + '_' + str(yprobs[i][j][0].year) + '-' + str(yprobs[i][j][-1].year) + '_probs.csv'
			np.savetxt(filename, outfile.T, delimiter =',', header = ','.join(labels))
	return()

#=============================================================================#
#===========================POST - PROCESSING=================================#
#=============================================================================#
rcp = [45,85]
models = ['HadGEM2-ES', 'HadGEM2-CC', 'HadGEM2-AO', 'CNRM-CM5', 'CMCC-CM', 'CESM1-CAM5', 'CESM1-BGC', 'CCSM4', 'CanESM2']
counts = [[],[]]
ooberror = 1-rfmodel.oob_score_
[sxfeature,sfeatureimp],[cmx,cmxperc],[fpr,tpr,thresholds,roc_auc] = rf_tests(fsampledata,rfmodel) #test the rfmodel
sn = tpr
sp = (fpr-1)/-1
d = np.abs(sn-sp)
thr = np.where(d==np.min(d))
threshhold = thresholds[thr] #this value may change with the new variable dataset
for r in range(len(rcp)):
	for m in models:
		counts[r].append(percent_series(yprobs, rcp[r], m, threshhold))
counts = np.array(counts)

baseline = len(yprobs_hi[0].data[yprobs_hi[0].data> threshhold])
t = np.arange(syear, eyear)
colors = ['orange', 'red']
plt.rcParams['figure.figsize'] =14,10
for i in range(2):
	plt.subplot(1,2,i+1)
	mu_model = np.mean(counts[i], axis =0)/baseline*100
	plt.plot(t,mu_model, 'red', lw = 4, label = 'Ensemble Average')
	for j in range(len(models)):
			modelprob = counts[i][j]/baseline*100
			plt.plot(t, modelprob, alpha = 0.5,lw = 1.5, label = models[j])
	plt.grid()
	plt.xlabel('Year')
	plt.ylabel('Percent Area (2010 baseline = %0.1f $km^2$)' %(baseline*.8))
	plt.legend(loc = 'upper right')
	plt.title('Suitable WBP bioclimate area; RCP ' + str(rcp[i])[0] +'.' + str(rcp[i])[1]+ '; threshold =' +str(np.round(threshhold,3)[0]))
plt.savefig('projections04182014.png', bbox_inches ='tight')

color = ['orange', 'red']
lab = ['RCP 4.5', 'RCP 8.5']
for i in range(2):
	std_model = np.std(counts[i], axis =0)/baseline*100
	plt.plot(t,std_model, 'red', lw = 4, label = lab[i], color = color[i])
	#for j in range(len(models)):
	#		modelprob = counts[i][j]/baseline*100
	#		plt.plot(t, modelprob, alpha = 0.5,lw = 1.5, label = models[j])
plt.grid()
plt.xlabel('Year')
plt.ylabel('$\sigma$ (Std Percent Area)')
plt.legend(loc = 'upper right')
plt.savefig('deviations04202014.png', bbox_inches ='tight')

#=============================================================================#
#============================BUILD ELEVATION TABLES===========================#
#=============================================================================#

def elevation_analysis(yprob, baselinearea=32031.2, threshhold=0.421, resolution=800): #threshhold is conservative
	aspect, slope, elev  = topo_extract()
	#h = header_extract()
	c = resolution / 1000
	#totalarea = h['ncols']*h['nrows']*c
	areainthresh = len(yprob[yprob>=threshhold])*c
	percentofarea = areainthresh/baselinearea
	meanelevation = np.mean(elev[yprob>=threshhold])
	quantile = np.percentile(elev[yprob>=threshhold], [2.5,97.5])
	output = [areainthresh, percentofarea, meanelevation, quantile[0],quantile[1]]
	return(output)	

def elev_summary(eaout, v):
	minv = [[],[]]
	maxv = [[],[]]
	avgv = [[],[]]
	#variables needed area, percent, mean elevation, 2.5 elevation, 97.5 elevation
	for r in range(len(eaout)):
		for y in range(len(eaout[0][0])):
			avgv[r].append(np.mean(to[r,:,y,v]))
			minv[r].append(np.min(to[r,:,y,v]))
			maxv[r].append(np.max(to[r,:,y,v]))
	return(np.array(avgv), np.array(minv), np.array(maxv))

'''
tab_out = [[],[]]
ytr = np.array([2010, 2040,2070,2099]) - syear
for r in range(len(rcp)):
	for m in range(len(models)):
		eas = []
		for y in ytr:
			ea_out = elevation_analysis(yprobs[m][r][y].data)
			eas.append(ea_out)
		tab_out[r].append(eas)
tab_out = np.array(tab_out)
to = np.array(tab_out)

#get the elevation summaries
#first find the areas
areas = np.round(elev_summary(to,0),1)
percent = np.round(elev_summary(to,1),3)*100
mu_elv = np.round(elev_summary(to,2),1)
min_elv = np.round(elev_summary(to,3),1)
max_elv = np.round(elev_summary(to,4),1)
'''
def print_line(values):
	i = 0
	line1 = print("& %0.1f & %0.1f & %0.1f & %0.1f\\" %(values[0][i][0],values[0][i][1],values[0][i][2],values[0][i][3]))
	line2 = print("&\it (%0.0f-%0.0f) & \it (%0.0f-%0.0f) & \it (%0.0f-%0.0f) & \it (%0.0f-%0.0f)\\" %(values[1][i][0],values[2][i][0],values[1][i][1],values[2][i][1],values[1][i][2],values[2][i][2],values[1][i][3],values[2][i][3])) 
	i = 1
	line3 = print("& %0.1f & %0.1f & %0.1f & %0.1f\\" %(values[0][i][0],values[0][i][1],values[0][i][2],values[0][i][3]))
	line4 = print("&\it (%0.0f-%0.0f) & \it (%0.0f-%0.0f) & \it (%0.0f-%0.0f) & \it (%0.0f-%0.0f)\\" %(values[1][i][0],values[2][i][0],values[1][i][1],values[2][i][1],values[1][i][2],values[2][i][2],values[1][i][3],values[2][i][3])) 
	return(line1,line2, line3, line4)
	
def print_table(to):
	areas = np.round(elev_summary(to,0),1)
	percent = np.round(elev_summary(to,1),3)*100
	mu_elv = np.round(elev_summary(to,2),1)
	min_elv = np.round(elev_summary(to,3),1)
	max_elv = np.round(elev_summary(to,3),1)
	out1 = print_line(areas)
	out2 = print_lines(percent)
	out3 = print_lines(mu_elv)
	out4 = print_lines(min_elv)
	out5 = print_lines(max_elv)
	
#find the high elevation value and the low one out of the gcms at each 30 year period

out = baseline
tab_out.append([out, out/baseline, 

#not complete....
'''
#=============================================================================#
#=========================SPATIALLY EXPLICIT PLOTS============================#
#=============================================================================#
ytr = np.array([2040,2070,2099]) - syear
for r in range(len(rcp)):
	for m in range(len(models)):
		for y in ytr:
			rf_plot(yprobs[m][r][y], save= 'y')
			plt.close()

ensavg_probs = [[],[]]
ele_out = [[],[]]
for r in range(len(rcp)):
	for y in ytr:
		c = 0
		temp = np.zeros(np.shape(yprobs[0][0][0].data))
		for m in range(len(models)):
			temp += yprobs[m][r][y].data
			c+=1
		edata = PGridData(year = yprobs[m][r][y].year, var = 'prob_pres', model = 'ensemble_avg', rcp = rcp[r], data = temp/c)
		ensavg_probs[r].append(edata)
		ele_out[r].append(elevation_analysis(edata.data))
		rf_plot(edata,save='y')
		plt.close()

##### Moving window average of presence count
'''

sampledata = np.genfromtxt('E:\\WBP_model\\fielddata\\1950_1980_merged_data.csv', delimiter = ',', names=True)
testdata = np.genfromtxt('E:\\WBP_model\\fielddata\\PRISM_1950_1980_data.csv', delimiter =',', names=True)
rfmodel = rf_fit(sampledata,8)
yp = rf_analysis(rfmodel, testdata)
p = PGridData(2010, 'prob_pres', 'PRISM', None, yp[1])
yprobs_hi = [p]
testyears = [2040,2070,2099]
models = ['CanESM2', 'CCSM4', 'CESM1-BGC', 'CESM1-CAM5', 'CMCC-CM', 'CNRM-CM5', 'HadGEM2-AO', 'HadGEM-CC', 'HadGEM2-ES']
rcps = [45,85]
timestart = time.clock()
for year in range(2040,2100):
	for m in models:
		for r in rcps:
			pfilename = 'E:\\WBP_model\\projections\\' + m + '\\' + m + '_'+ str(r) + '_' + str(year-30)+ '_' + str(year) + '_data.csv'
			projectiondata = np.genfromtxt(pfilename, delimiter =',', names = True)
			#fprojectiondata = CovariateFilter(projectiondata, flist)
			yp = rf_analysis(rfmodel, projectiondata)
			p = PGridData(year, 'prob_pres', m, r, yp[1])
			yprobs_hi.append(p)
			draw_progress_bar(year/(2100-2040))
timeend = time.clock()
count =[]
for i in range(len(yprobs_hi)):
	count.append(len(yprobs_hi[i].data[yprobs_hi[i].data>= 0.4]))
count = np.array(count)
t = np.arange(2040,2100)
plt.plot(t,count[1:])
plt.ylabel('$km^2$ presence')
plt.xlabel('year')
plt.grid()
'''
