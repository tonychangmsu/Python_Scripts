import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap

def filterSpecies(fiadata,cde):
#provide fia plot data and species code desired and returns queried fia data
	return(fiadata[(fiadata['SPCD'] == cde)])
	
def filterArea(fiadata, AOA):
#provide fia plot data and AOA as a list = [xmin, ymin, xmax, ymax]
	return(fiadata[(fiadata.LON > AOA[0]) & (fiadata.LON < AOA[2]) & (fiadata.LAT > AOA[1]) & (fiadata.LAT < AOA[3])])

def getAbsence(fiadata, out_cde):
#provide fia plot data and species code to be excluded
	return(fiadata[(fiadata['SPCD'] != out_cde)])

def linkCN(fiadata, plot_table):
#links the fiadata to the reference plot_table by the PLT_CN attribute
	return(plot_table[plot_table.CN.isin(fiadata.PLT_CN)])

def plotDistribution(data, bins = 30):
	#enter the data of interest and plots a normal distribution curve and histogram of the data
	count, bins = np.histogram(data, bins=30, normed=True)
	mu, sigma = data.mean(), data.std() #solve the mean/sd for elevation
	data.hist(bins=bins, alpha=0.5, normed=True)
	plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2)
	return()
	
if __name__ == '__main__':
	#Constants
	
	xmax = -108.263; xmin = -112.436; ymin = 42.252; ymax = 46.182 # GYE bounds
	AOA = [xmin, ymin, xmax, ymax] #specify the bounds for the FIA data
	
	#AOA = [-117.237, 31.343, -102.037,49.00] #LCC bounds
	
	whitebark_code = 101
	adultDBH = 7.87
	limber_code = 113

	workspace = "E:\\FIA\\"
	#get the plot data  
	#plots provide the lat,lon, and elevation data
	pfname ="PLOT.CSV"
	plotfilename = "%s%s" %(workspace,pfname)
	plots = pd.read_csv(plotfilename)
	
	#now get the seedling data
	fname ="SEEDLING.CSV"
	seedfilename = "%s%s" %(workspace,fname)
	seed = pd.read_csv(seedfilename)

	#filter out whitebark pine: 101
	wbpseed = filterSpecies(seed, whitebark_code)
	#try for limber pine: 113
	lbpseed = filterSpecies(seed, limber_code)
	
	#get tree data
	sname = "WBPTREE.CSV"
	treefilename = "%s%s" %(workspace, sname)
	trees = pd.read_csv(treefilename, low_memory=False)
	wbpadult = trees[trees.DIA>=adultDBH]
	
	'''	
	#wbptree = filterSpecies(trees, whitebark_code)
	#write out the filtered dataframe to reduce memory use
	#wbptree.to_csv('E:\\FIA\\WBPTREE.CSV') #written out 08/25/2014 @t.chang
	'''
	
	#link the tree data with the plots
	lwbp =  linkCN(wbpadult, plots)
	wbp_adult_gye = filterArea(lwbp, AOA)
	
	#find all the matching plots CNs
	wbp_match_plots = linkCN(wbpseed, plots)
	lbp_match_plots = linkCN(lbpseed, plots)

	#filter within some bounds (end product here!)
	wbp_seed_gye = filterArea(wbp_match_plots, AOA)
	lbp_seed_gye = filterArea(lbp_match_plots, AOA)

	#need absences too, so consider all the seedling plots as an absence
	wbp_abs = getAbsence(seed, whitebark_code)
	awbp_link = linkCN(wbp_abs, plots)
	wbp_abs_gye = filterArea(awbp_link, AOA)
	
	#create a data file of presences and absences
	wbp_seed_gye.insert(0, 'p_a', np.ones(len(wbp_seed_gye))) #add a column for presence
	pres = wbp_seed_gye[['p_a', 'LAT', 'LON', 'ELEV']]
	wbp_abs_gye.insert(0, 'p_a', np.zeros(len(wbp_abs_gye)))
	abse = wbp_abs_gye[['p_a', 'LAT', 'LON', 'ELEV']]
	wbpdata = pres.append(abse)
	#wbpdata.to_csv('E:\\FIA\\WBP_SEED_PA.CSV') #written out 08/25/2014 @t.chang
	
	'''
	First major question:
	- compare adults to seedling distributions
	- do they differ by elevation
	'''
	#question number 1
	plotDistribution(wbp_adult_gye.ELEV);plotDistribution(wbp_seed_gye.ELEV)
	t,p = stats.ttest_ind(wbp_adult_gye.ELEV, wbp_seed_gye.ELEV, equal_var = False)
	print(p)
	
	'''
	- yes, it appears that there is a significant difference in the mean
	- however, by the inspection of the quantiles, not by much
	'''

	'''
	Second major question:
	- are the distribution curves sensitive to misclassification of limber pine?

	#compare distributions
		plotDistribution(wbp_seed_gye.ELEV); plotDistribution(lbp_seed_gye.ELEV)
		plotDistribution(wbp_seed_gye.ELEV); plotDistribution(wbp_abs_gye.ELEV)
	
	- we will want to consider the how elevation zones differ for seedlings from adults and perform a t-test.
	- we can consider wbp_seed versus lbp_seeds first.
		
		t, p = stats.ttest_ind(wbp_seed_gye.ELEV, lbp_seed_gye.ELEV, equal_var = False)
		# p-value very small... although lbp does not seem like it is normally distributed

	Next:
	- Should we incorporate lbp from different elevations into the model? We could perform a sensitivity analysis using the following criteria
	
		1st case: none lbp are misidentified so we do not incorporate any lbp into the seedling dataset
		2nd case: some of the lbp are misidentified so we incorporate the upper 75% of lbp elevations into the dataset (> 8295.0ft)
		3rd case: some of the lbp are misidentified so we incorporate the upper 50% of lbp elevations into the dataset (> 7865.0ft)
		4th case: some of the lbp are misidentified so we incorporate the upper 25% of lbp elevations into the dataset (> 7865.0ft)
		5th case: all lbp are misidentified so we incorporate all lbp into the seedling dataset
	
	So lets do this first...
	'''
	
	cutoff = [75,50,25]
	# 2nd case
	lbp_75 = np.percentile(lbp_seed_gye.ELEV, [cutoff[0],100])[0]
	wbp_seed_2nd = wbp_seed_gye.append(lbp_seed_gye[lbp_seed_gye.ELEV>=lbp_75])
	# 3rd case
	lbp_50 = np.percentile(lbp_seed_gye.ELEV, [cutoff[1],100])[0]
	wbp_seed_3rd = wbp_seed_gye.append(lbp_seed_gye[lbp_seed_gye.ELEV>=lbp_50])
	# 4th case
	lbp_25 = np.percentile(lbp_seed_gye.ELEV, [cutoff[2],100])[0]
	wbp_seed_4th = wbp_seed_gye.append(lbp_seed_gye[lbp_seed_gye.ELEV>=lbp_25])
	# 5th case
	wbp_seed_expand = wbp_seed_gye.append(lbp_seed_gye)
	
	# how does this change the distributions?
	plotDistribution(wbp_seed_gye.ELEV);plotDistribution(wbp_seed_2nd.ELEV);plotDistribution(wbp_seed_3rd.ELEV);plotDistribution(wbp_seed_4th.ELEV);plotDistribution(wbp_seed_expand.ELEV);
	# not much 

	'''
	- we can go about this in a manner that is more heuristic/iterative 
	- better method to check by
	- still "cheating"
	'''	
	
	# plot where significant change occurs
	tests = pd.DataFrame(columns = ['perc', 'elev_th', 'p_val'])
	alpha = 0.05 #level of type I error acceptable
	
	for i in range(100):
		lbp_add = np.percentile(lbp_seed_gye.ELEV, [i,100])[0]
		wbp_seed_add = wbp_seed_gye.append(lbp_seed_gye[lbp_seed_gye.ELEV>=lbp_add])
		t, p = stats.ttest_ind(wbp_seed_gye.ELEV, wbp_seed_add.ELEV, equal_var = False)
		idata = [i, lbp_add, p]
		tests.loc[i] = idata
		
	plt.plot(tests.elev_th, tests.p_val); plt.plot(tests.elev_th,np.ones(len(tests))*alpha)
	#find the first instance where p-value is below 0.05
	percentile_cut = tests[tests.p_val<=0.05].perc.iloc[-1] #returns 41, pretty sneaky...
	
	'''
	At this point we can gather all the environmental variables for the seedling data and then perform the Random Forest analysis again
	for the reference period of 1980-2010. This should work well, since we have a high prevalence value?

	What we also need to consider is the co-occurance of other trees.
	
	Link to environmental variables to be finished today 08.27.2014 @t.chang
	
	'''
	'''
	compare to Nate's list...
		cleaned = pd.read_csv('E:\\FIA\\fia_obs_clean4.10.2014.csv')
		nwbp = filterSpecies(cleaned, whitebark_code)
		fwbp = filterArea(nwbp, AOA)
		uwbp = fwbp.drop_duplicates('CN')
	find common CN with FIA PLOTS
		match = fwbp[fwbp.CN.isin(uwbp.CN)]
	plot US/practice with basemap...?

	i = 1
	m = Basemap(llcrnrlon=xmin-i,llcrnrlat=ymin-i,urcrnrlon=xmax+i,urcrnrlat=ymax+i,projection='tmerc',
				resolution='i', lon_0 = xmin+(xmax-xmin)/2, lat_0 = ymin+(ymax-ymin)/2)
	m.shadedrelief()
	m.readshapefile('D:\CHANG\GIS_Data\GYE_shapes\GYE', 'points')
	m.readshapefile('D:\CHANG\GIS_Data\GYE_shapes\YELL_GRTE', 'points')
	wbpx,wbpy = m(wbp_gye.LON, wbp_gye.LAT)
	lbpx, lbpy = m(lbp_gye.LON, lbp_gye.LAT)
	m.scatter(wbpx, wbpy,3,marker='o',color='g')
	m.scatter(lbpx, lbpy,3,marker='x',color='r')
	'''

