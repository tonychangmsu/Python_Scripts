#Title: WBP_adult_seed_analysis.py
#Author: Tony Chang
#Abstract: Compares the climate values between the linked adult/seedling wbp present in GYE
#Date: 10/16/2014

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plotDistribution(data, bins = 30):
	#enter the data of interest and plots a normal distribution curve and histogram of the data
	count, bins = np.histogram(data, bins=30, normed=True)
	mu, sigma = data.mean(), data.std() #solve the mean/sd for elevation
	data.hist(bins=bins, alpha=0.5, normed=True)
	plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2)
	return()

def compareDistributions(data1, data2, labels = ['data1', 'data2']):
	#generates histograms for the 12 month climate variables for 2 different datasets defined by their labels
	colors = ['blue', 'red']
	z = [data1,data2]
	c = 0
	for data in z:
		count, bins = np.histogram(data, bins=30, normed=True)
		mu, sigma = data.mean(), data.std() #solve the mean/sd for elevation
		data.hist(bins=bins, alpha=0.5, normed=True, color = colors[c])
		plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color = colors[c], label =labels[c])
		c+=1
	return()

def factor_scatter_matrix(df, factor, palette=None):
    '''Create a scatter matrix of the variables in df, with differently colored
    points depending on the value of df[factor].
    inputs:
        df: pandas.DataFrame containing the columns to be plotted, as well 
            as factor.
        factor: string or pandas.Series. The column indicating which group 
            each row belongs to.
        palette: A list of hex codes, at least as long as the number of groups.
            If omitted, a predefined palette will be used, but it only includes
            9 groups.
    '''
    import matplotlib.colors
    import numpy as np
    from pandas.tools.plotting import scatter_matrix
    from scipy.stats import gaussian_kde

    if isinstance(factor,str):
        factor_name = factor #save off the name
        factor = df[factor] #extract column
        df = df.drop(factor_name,axis=1) # remove from df, so it 
        # doesn't get a row and col in the plot.

    classes = list(set(factor))

    if palette is None:
        palette = ['#e41a1c', '#377eb8', '#4eae4b', 
                   '#994fa1', '#ff8101', '#fdfc33', 
                   '#a8572c', '#f482be', '#999999']

    color_map = dict(zip(classes,palette))

    if len(classes) > len(palette):
        raise ValueError('''Too many groups for the number of colors provided. We only have {} colors in the palette, but you have {} groups.'''.format(len(palette), len(classes)))

    colors = factor.apply(lambda group: color_map[group])
    axarr = scatter_matrix(df,figsize=(10,10),marker='o',c=colors,diagonal=None)

    for rc in range(len(df.columns)):
        for group in classes:
            y = df[factor == group].icol(rc).values
            gkde = gaussian_kde(y)
            ind = np.linspace(y.min(), y.max(), 1000)
            axarr[rc][rc].plot(ind, gkde.evaluate(ind),c=color_map[group])
    return(axarr, color_map)
	
if __name__ == '__main__':
	#get the datasets
	wbp_adult = pd.read_csv("E:\\WBP_model\\New_Analysis\\FIA_ADULT_merged_cleaned.csv")
	wbp_seed = pd.read_csv("E:\\WBP_model\\New_Analysis\\FIA_SEEDMIX_merged_cleaned.csv") 
	p_adult = wbp_adult[wbp_adult.response==1]
	p_seed = wbp_seed[wbp_seed.response!=0]
	labels = ['adult', 'seedlings']
	v = 'tmax'
	for i in range(1,13):
		#test difference in pet and aet
		var = '%s%i'%(v,i)
		plt.subplot(3,4,i)
		#data1 = p_adult['pet%i'%i]-p_adult['aet%i'%i]
		#data2 = p_seed['pet%i'%i]-p_seed['aet%i'%i]
		data1 = p_adult[var]
		data2 = p_seed[var]
		compareDistributions(data1,data2, labels)
		plt.title('%s'%(var))
		plt.legend()
	plt.savefig('%s_compare.png'%v)
	#no major difference between any of the distributions (as expected)
	# we can check if there is a difference with the different climate periods using the Piekielek dataset
	wbp_adult_pie = pd.read_csv('E:\\WBP_model\\fielddata\\1950_1980_merged_data.csv')
	p_adult2 = wbp_adult_pie[wbp_adult_pie.response==1]
	#the results are radically different due to climate in 1950-1980 being substantially warmer than in the past
	"""
	code for testing water deficit
	for i in range(1,13):
		#test difference in pet and aet
		v = 'water deficit'
		plt.subplot(3,4,i)
		data1 = p_adult['pet%i'%i]-p_adult['aet%i'%i]
		data2 = p_seed['pet%i'%i]-p_seed['aet%i'%i]
		compareDistributions(data1,data2,labels)
		plt.title('%s for month %i'%(v,i))
		plt.legend()
	plt.savefig('Deficit_compare.png')
	"""
	
	t,p = stats.ttest_ind(p_adult.pet8-p_adult.aet8, p_seed.pet8-p_seed.aet8, equal_var = False)
	
	#consider the scatter matrix by group
	#change the adult presence to class 3?
	wbp_adult.response = wbp_adult.response*3
	wbp_grouped = wbp_seed.append(wbp_adult)
	wbp_sub = wbp_grouped[['response', 'tmax8', 'tmin1', 'pack5', 'pack9']]
	factor_scatter_matrix(wbp_sub[wbp_sub.response!=0], 'response')
