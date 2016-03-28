import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.stats as stats
#from mpl_toolkits.basemap import Basemap

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
	
	#AOA = [-117.237, 31.343, -102.037,49.00] #GNLCC bounds
	
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
	#consider using the abundances
	wbpgroup = wbpseed.groupby('PLT_CN')
	wbp_abun = wbpgroup.aggregate({'TREECOUNT': np.sum})
	
	#try for limber pine: 113
	lbpseed = filterSpecies(seed, limber_code)
	lbpgroup = lbpseed.groupby('PLT_CN')
	lbp_abun = lbpgroup.aggregate({'TREECOUNT': np.sum})
	
	#get tree data
	sname = "WBPTREE.CSV"
	wbptreefilename = "%s%s" %(workspace, sname)
	wbptrees = pd.read_csv(wbptreefilename, low_memory=False)
	wbpadult = wbptrees[wbptrees.DIA>=adultDBH]
	
	lname = "LBPTREE.CSV"
	lbptreefilename = "%s%s" %(workspace, lname)
	lbptrees = pd.read_csv(lbptreefilename, low_memory=False)
	lbpadult = lbptrees[lbptrees.DIA>=adultDBH]
	
	'''	
	sname = 'TREE.CSV'
	treefilename = "%s%s" %(workspace, sname)
	trees = pd.read_csv(treefilename)
	
	#wbptree = filterSpecies(trees, whitebark_code)
	#write out the filtered dataframe to reduce memory use
	#wbptree.to_csv('E:\\FIA\\WBPTREE.CSV') #written out 08/25/2014 @t.chang
	#get limber code too
	
	lbptree = filterSpecies(trees, limber_code)
	lbptree.to_csv('E:\\FIA\\LBPTREE.CSV')
	'''
	
	#link the tree data with the plots
	lwbp =  linkCN(wbpadult, plots)
	wbp_adult_gye = filterArea(lwbp, AOA)
	lbp_adult_gye = filterArea(linkCN(lbpadult,plots), AOA)
	
	#create an adult file of Presence/Absence
	flist = ['CN', 'PLOT','LAT','LON', 'ELEV']
	reduced_wbp_adult_gye = wbp_adult_gye[flist]
	
	#find all the matching plots CNs
	wbp_match_plots = linkCN(wbpseed, plots)
	lbp_match_plots = linkCN(lbpseed, plots)
	wbp_abun_match = pd.merge(plots, wbp_abun, left_on='CN', right_index=True) #merge the sets together
	lbp_abun_match = pd.merge(plots, lbp_abun, left_on='CN', right_index=True)
	
	#filter within some bounds (end product here!)
	wbp_seed_gye = filterArea(wbp_match_plots, AOA)
	lbp_seed_gye = filterArea(lbp_match_plots, AOA)
	wbp_aseed_gye = filterArea(wbp_abun_match, AOA)
	lbp_aseed_gye = filterArea(lbp_abun_match, AOA)
	
	#Plot the abundances to see how they are distributed by elevation
	plt.scatter(wbp_aseed_gye.ELEV, wbp_aseed_gye.TREECOUNT)
	plt.scatter(lbp_aseed_gye.ELEV, lbp_aseed_gye.TREECOUNT)
	
	slope, intercept, r_value, p_value, std_err = stats.linregress(wbp_aseed_gye[~np.isnan(wbp_aseed_gye.TREECOUNT)].ELEV, wbp_aseed_gye[~np.isnan(wbp_aseed_gye.TREECOUNT)].TREECOUNT)
	'''
	no apparent relationship with elevation and abundance.
	p-value : 0.2825
	slope : 0.00098
	r-value : 0.08
	
	Furthermore, there appears to be a large disparity of abundance counts 
	
	len(wbp_aseed_gye[~np.isnan(wbp_aseed_gye.TREECOUNT)])
	
	(only 178 samples within the GYE to do anything with..)
	
	plt.scatter(wbp_aseed_gye.LON, wbp_aseed_gye.LAT, color = 'red', label = 'wbp seedling presence')
	plt.scatter(wbp_aseed_gye[~np.isnan(wbp_aseed_gye.TREECOUNT)].LON, wbp_aseed_gye[~np.isnan(wbp_aseed_gye.TREECOUNT)].LAT, color = 'green', label = 'wbp seedling abundance record')
	plt.legend()
	
	at this point, I would argue that we do not have nearly enough samples to generate spatially explicit inference regarding wbp seedling abundance for the GYE range. 
	
	we might consider expanding the spatial extent (using the GNLCC), but this would require some extensive work in regards to rebuilding the dataset for water balance and other associated climate variables. 
	
	#AOA = [-117.237, 31.343, -102.037,49.00] #GNLCC bounds
	#len(wbp_aseed_gye[~np.isnan(wbp_aseed_gye.TREECOUNT)])
	
	This returns 745 points, but for a much larger spatial extent (almost 8 times bigger). In the end, we could just perform the analysis anyways, but must be explicit that we are under represented for most of the range. 
	
	Before we stop here, we might consider adult abundances and compare the seedling group to the adult group
	'''
	'''
	sname = 'TREE.CSV'
	treefilename = "%s%s" %(workspace, sname)
	trees = pd.read_csv(treefilename)
	
	# first we would like to get some information regarding wbp adult tree counts in general. 
	# While we are at it, we might as well gather information on the other sets of trees that include:
	# lodgepole pine, douglas fir, subalpine fir, engelmann spruce, and aspen
	# Could develop a master list for these specified trees to reduce the overall size of the TREE dataset
	'''
	"""
	pial_code = spcd[spcd.COMMON_NAME=='whitebark pine'].SPCD
	pifl_code = spcd[spcd.COMMON_NAME=='limber pine'].SPCD
	abla_code = spcd[spcd.COMMON_NAME=='subalpine fir'].SPCD
	pico_code = spcd[spcd.COMMON_NAME=='lodgepole pine'].SPCD
	potr_code = spcd[spcd.COMMON_NAME=='quaking aspen'].SPCD
	pien_code = spcd[spcd.COMMON_NAME=='Engelmann spruce'].SPCD
	psme_code = spcd[spcd.COMMON_NAME=='Douglas-fir'].SPCD
	jusc_code = spcd[spcd.COMMON_NAME=='Rocky Mountain juniper'].SPCD
	"""
	'''
	pial_code = 101
	pifl_code = 113
	abla_code = 19
	pico_code = 108
	potr_code = 746
	pien_code = 93
	psme_code = 202
	jusc_code = 66
	tree_list = np.array([pial_code, pifl_code, abla_code, pico_code, potr_code, pien_code, psme_code, jusc_code])
	#filter the tree list to these species
	GYE_trees = trees[trees.SPCD.isin(tree_list)]
	
	#write out the culled dataset
	#GYE_trees.to_csv('E:\\FIA\\GYETREES.CSV') #produced 08/29/2014 @t.chang
	#need absences too, so consider all the seedling plots as an absence
	'''
	GYE_trees = pd.read_csv('E:\\FIA\\GYETREES.CSV')
	# Link GYE_trees with lat, lon and elevation?
	trees_linked = pd.merge(plots, GYE_trees, left_on='CN', right_on='PLT_CN')
	
	tree_reduced = trees_linked[['PLT_CN','SPCD','DIA','LAT','LON', 'ELEV']]

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
	
	#generate a combined lbp and wbp seedling presence and absence count
	lbp_seed_gye.insert(0, 'p_a', np.ones(len(lbp_seed_gye))*2) #add a column for lbp presence
	wbp_lbp_seed_gye = wbp_seed_gye.append(lbp_seed_gye)
	lwbp_pres = wbp_lbp_seed_gye[['p_a', 'LAT', 'LON', 'ELEV']]
	lwbpdata = lwbp_pres.append(abse)
	lwbpdata.to_csv('E:\\FIA\\WBP_SEEDMIX_PA.CSV') #written out 10/17/2014 @t.chang
	