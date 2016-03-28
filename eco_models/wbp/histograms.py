#histogram plots for presence and absence

import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import sklearn
import gdal
from gdalconst import *
import osr 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from scipy import stats as sp

sampledata = np.genfromtxt('E:\\WBP_model\\fielddata\\1950_1980_merged_data.csv', delimiter = ',', names=True)
testdata = np.genfromtxt('E:\\WBP_model\\fielddata\\PRISM_1950_1980_data.csv', delimiter =',', names=True)
flist = ['tmax7', 'pack4', 'vpd3', 'ppt9', 'tmin1', 'aet7', 'ppt4', 'pet8'] 
y_rp = sampledata['response']
fsampledata = covariate_filter(sampledata, flist, 'y')
ftestdata = covariate_filter(testdata, flist)
rfmodel = rf_fit(fsampledata)
yp = rf_analysis(rfmodel, ftestdata)
p = PGridData(2010, 'prob_pres', 'PRISM', None, yp[1])
yprobs_hi = [p]
p_pr = np.where(yprobs_hi[0].data >0.421)
a_pr = np.where(yprobs_hi[0].data <=0.421)
testtmax = np.reshape(ftestdata['tmax7'], np.shape(yprobs_hi[0].data))
testpack = np.reshape(ftestdata['pack4'], np.shape(yprobs_hi[0].data))
tx = np.linspace(np.min(testtmax), np.max(testtmax), 1000)
sx = np.linspace(np.min(testpack), np.max(testpack), 1000)
plt.subplot(1,2,1)

#plt.plot(tx, sp.norm.pdf(tx, np.median(testtmax[p_pr]), np.std(testtmax[p_pr])), color = 'blue', lw =2)
#plt.plot(tx, sp.norm.pdf(tx, np.median(testtmax[a_pr]), np.std(testtmax[a_pr])), color = 'orange', lw =2)
plt.hist(testtmax[p_pr], bins = bins, color = 'blue', normed = 1, alpha = 0.5)
b = 25
ctmax, btmax = np.histogram(testtmax[p_pr], normed = True, bins = b)
actmax, abtmax = np.histogram(testtmax[a_pr], normed = True, bins = b)
#plt.plot(tx, sp.norm.pdf(tx, np.median(testtmax[a_pr]), np.std(testtmax[a_pr])), color = 'orange', lw =2)
#plt.hist(testtmax[a_pr], bins = bins, color = 'orange', normed = 1, alpha = 0.5)
#plt.subplot(1,2,2)
b = 27
cpack, bpack = np.histogram(testpack[p_pr], normed = True, bins = b)
acpack, abpack = np.histogram(testpack[a_pr], normed = True, bins = b)

#plt.plot(sx, sp.norm.pdf(sx, np.median(testpack[p_pr]), np.std(testpack[p_pr])), color = 'blue', lw =2)
#plt.plot(sx, sp.norm.pdf(sx, np.median(testpack[a_pr]), np.std(testpack[a_pr])), color = 'orange', lw =2)

a_i = np.where(y_rp==0)
p_i = np.where(y_rp==1)

vars = ['tmax7', 'pack4']
plt.rcParams['figure.figsize'] = 14,5
i = 0
plt.subplot(1,2,i+1)
bins = 16
weights = np.ones_like(sampledata[vars[i]][a_i])/len(sampledata[vars[i]][a_i])
#plt.hist(sampledata[vars[i]][a_i], weights=weights, bins =40, color = 'red', alpha = 0.6, label = 'absence ($\mu = %0.1f , \sigma = %0.1f$)' %(np.mean(sampledata[vars[i]][a_i]), np.std(sampledata[vars[i]][a_i])))
plt.hist(sampledata[vars[i]][a_i], weights=weights, bins =bins, color = 'red', alpha = 0.6, label = 'absence ($\mu = %0.1f$)' %(np.mean(sampledata[vars[i]][a_i])))
plt.vlines(np.mean(sampledata[vars[i]][a_i]), 0, 0.2, color = 'red', linestyle = '--', linewidth = 2)
weights = np.ones_like(sampledata[vars[i]][p_i])/len(sampledata[vars[i]][p_i])
#plt.hist(sampledata[vars[i]][p_i], weights=weights, bins = 40, color = 'green', alpha = 0.6, label = 'presence ($\mu = %0.1f , \sigma = %0.1f$)' %(np.mean(sampledata[vars[i]][p_i]), np.std(sampledata[vars[i]][p_i])))
plt.hist(sampledata[vars[i]][p_i], weights=weights, bins = bins, color = 'green', alpha = 0.6, label = 'presence ($\mu = %0.1f$)' %(np.mean(sampledata[vars[i]][p_i])))
plt.vlines(np.mean(sampledata[vars[i]][p_i]), 0, .2, color = 'green', linestyle = '--', linewidth = 2)
#plt.plot(tx, sp.norm.pdf(tx, np.median(testtmax[p_pr]), np.std(testtmax[p_pr])), color = 'blue', lw =2, label = 'RF modeled presence')
plt.plot(btmax[:-1], 0.9*ctmax, color = 'blue', lw =2, label = 'RF modeled presence')
#plt.plot(abtmax[:-1], 0.9*actmax, color = 'blue', lw =2, label = 'RF modeled presence')
plt.xlabel('tmax7 ($^oC$)')
plt.ylabel('Normalized Density')
plt.grid()
plt.legend()

i = 1
plt.subplot(1,2,i+1)
bins = 27
weights = np.ones_like(sampledata[vars[i]][a_i])/len(sampledata[vars[i]][a_i])
#plt.hist(sampledata[vars[i]][a_i], weights=weights, bins =40, color = 'red', alpha = 0.6, label = 'absence ($\mu = %0.1f , \sigma = %0.1f$)' %(np.mean(sampledata[vars[i]][a_i]), np.std(sampledata[vars[i]][a_i])))
plt.hist(sampledata[vars[i]][a_i], weights=weights, bins =bins, color = 'red', alpha = 0.6, label = 'absence ($\mu = %0.1f$)' %(np.mean(sampledata[vars[i]][a_i])))
plt.vlines(np.mean(sampledata[vars[i]][a_i]), 0, 0.14, color = 'red', linestyle = '--', linewidth = 2)
weights = np.ones_like(sampledata[vars[i]][p_i])/len(sampledata[vars[i]][p_i])
#plt.hist(sampledata[vars[i]][p_i], weights=weights, bins = 40, color = 'green', alpha = 0.6, label = 'presence ($\mu = %0.1f , \sigma = %0.1f$)' %(np.mean(sampledata[vars[i]][p_i]), np.std(sampledata[vars[i]][p_i])))
plt.hist(sampledata[vars[i]][p_i], weights=weights, bins =bins, color = 'green', alpha = 0.6, label = 'presence ($\mu = %0.1f$)' %(np.mean(sampledata[vars[i]][p_i])))
plt.vlines(np.mean(sampledata[vars[i]][p_i]), 0, .14, color = 'green', linestyle = '--', linewidth = 2)
#plt.plot(sx, 70*sp.norm.pdf(sx, np.median(testpack[p_pr]), np.std(testpack[p_pr])), color = 'blue', lw =2, label = 'RF modeled presence')
#plt.plot(sx, 70*sp.norm.pdf(sx, np.median(testpack[a_pr]), np.std(testpack[a_pr])), color = 'orange', lw =2)
plt.plot(bpack[:-1], 60*cpack, color = 'blue', lw =2,label = 'RF modeled presence')
plt.xlabel('pack4 ($mm$)')
plt.ylabel('Normalized Density')
plt.grid()
plt.legend()

plt.savefig('histo.png', bbox_inches= 'tight')