#Title: WBP_adult_seed_analysis.py
#Author: Tony Chang
#Abstract: Compares the climate values between the linked adult/seedling wbp present in GYE
#Date: 10/16/2014

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import date

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
	m = []
	s = []
	for data in z:
		count, bins = np.histogram(data, bins=30, normed=True)
		mu = data.mean()
		sigma = data.std() #solve the mean/sd 
		m.append(mu)
		s.append(sigma)
		data.hist(bins=bins, alpha=0.5, normed=True, color = colors[c])
		ax =plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color = colors[c], label = '%s ($\mu = %.1f, \sigma = %.1f$)'%(labels[c],mu, sigma))
		c+=1
	return(m,s)

def boxAndWhiskers(data1, data2, labels):
	def setBoxColors(bp, a = 0.7, lw =2, fs=10):
		setp(bp['boxes'][0], color='blue', alpha=a, lw=lw)
		setp(bp['caps'][0], color='blue', alpha=a, lw=lw)
		setp(bp['caps'][1], color='blue', alpha=a, lw=lw)
		setp(bp['whiskers'][0], color='blue', alpha=a, lw=lw)
		setp(bp['whiskers'][1], color='blue', alpha=a, lw=lw)
		setp(bp['fliers'][0], color='blue', marker = '.', markersize=fs, alpha=a)
		setp(bp['fliers'][1], color='blue', marker = '.', markersize=fs, alpha=a)
		setp(bp['medians'][0], color='black', alpha=a, lw=lw)

		setp(bp['boxes'][1], color='red', alpha=a, lw=lw)
		setp(bp['caps'][2], color='red', alpha=a, lw=lw)
		setp(bp['caps'][3], color='red', alpha=a, lw=lw)
		setp(bp['whiskers'][2], color='red', alpha=a, lw=lw)
		setp(bp['whiskers'][3], color='red', alpha=a, lw=lw)
		setp(bp['fliers'][2], color='red', marker = '.', markersize=fs, alpha=a)
		setp(bp['fliers'][3], color='red', marker = '.', markersize=fs, alpha=a)
		setp(bp['medians'][1], color='black', alpha=a, lw=lw)
		return()
	z = [data1,data2]
	bp = plt.boxplot(z, widths = 0.4)
	#bp = plt.boxplot(z)
	setBoxColors(bp)
	plt.xticks([1,2],labels)
	plt.grid()
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
	plt.rcParams['figure.figsize'] = 18,16
	#get the datasets
	wbp_adult = pd.read_csv("E:\\WBP_model\\New_Analysis\\FIA_ADULT_merged_cleaned.csv")
	wbp_seed = pd.read_csv("E:\\WBP_model\\New_Analysis\\FIA_SEEDMIX_merged_cleaned.csv") 
	p_adult = wbp_adult[wbp_adult.response==1] 
	p_seed = wbp_seed[wbp_seed.response==1]
	#add the variable tmean
	for i in range(1,13):
		var_name = 'tmean%i'%i
		adult_mean = (p_adult['tmax%i'%i]+p_adult['tmin%i'%i])/2
		seed_mean = (p_seed['tmax%i'%i]+p_seed['tmin%i'%i])/2
		p_adult[var_name] = pd.Series(adult_mean, index = p_adult.index)
		p_seed[var_name] = pd.Series(seed_mean, index = p_seed.index)
	adult_annual = np.mean(p_adult.iloc[:,-12:], axis=1)
	seed_annual = np.mean(p_seed.iloc[:,-12:], axis=1)
	#add the annual mean temperatures 
	p_adult['tann'] = pd.Series(adult_annual, index = p_adult.index)
	p_seed['tann'] = pd.Series(seed_annual, index = p_seed.index)
	
#Jacobs and Weaver analysis of photosynthesis rates
#additional analysis for Katie Ireland @t.chang 11/07/2014
mean_data_seed = (p_seed.tmean7+p_seed.tmean8)/2
plt.rcParams['figure.figsize'] = 10,8
fig = plt.figure()
ax = fig.add_subplot(111)
count, bins = np.histogram(mean_data_seed, bins=30, normed=True)
mu, sigma = mean_data_seed.mean(), mean_data_seed.std() #solve the mean/sd for elevation
mean_data_seed.hist(bins=bins, alpha=0.5, normed=True)
p2 = ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color = 'blue', label = 'FIA plots')
for tl in ax.get_yticklabels():
	tl.set_color('b')
ax2 = ax.twinx()

temp_data = [0,15,25,35]
A_data = [0.19,0.28,0.2,0.15]
A_data2 = [0.38,0.7,0.63,0.3]
p1 = ax2.plot(temp_data,A_data, color = 'red', ls = '--', label = 'Preconditioned D15-N5')
#ax2.plot(temp_data,A_data2, color = 'green', ls = '--', label = 'Preconditioned D15-N5')
ax.set_ylabel('Normalized frequency', color = 'b')
ax2.set_ylabel('Photosynthesis ($\mu m \cdot CO_2 m^{-2}s^{-1}$)', color = 'r')
for tl in ax2.get_yticklabels():
	tl.set_color('r')
ax.set_xlabel('Temperature ($^oC$)',)
lns = p1 + p2
ax.legend(lns, ['Preconditioned D15-N5', 'FIA plots'], loc = 0)
plt.savefig('E:\\Data_requests\\ireland_11072014.png', bbox_inches='tight')
	'''
	labels = ['adult', 'seedlings']
	variables = ['ELEV', 'tann', 'tmax7', 'tmin1', 'pack4']
	units = ['$ft$', '$^oC$', '$^oC$', '$^oC$', '$mm$']
	i = 1
	for var in range(len(variables)):
		#test difference in pet and aet
		#var = '%s%i'%(v,i)
		ax = plt.subplot(5,2,i)
		plt.xlabel(units[var])
		plt.ylabel('Normalized frequency')
		#data1 = p_adult['pet%i'%i]-p_adult['aet%i'%i]
		#data2 = p_seed['pet%i'%i]-p_seed['aet%i'%i]
		data1 = p_adult[variables[var]]
		data2 = p_seed[variables[var]]
		m,s = compareDistributions(data1,data2, labels)
		plt.title('%s'%(variables[var]))
		box = ax.get_position()
		ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
		# Put a legend below current axis
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=False, ncol=5)
		
		i +=1
		ax2 = plt.subplot(5,2,i)
		plt.title('%s'%(variables[var]))
		boxAndWhiskers(data1,data2,labels)
		plt.ylabel(units[var])
		confidence = 0.95
		q1 = stats.norm.interval(confidence, loc=m[0], scale=s[0])
		q2 = stats.norm.interval(confidence, loc=m[1], scale=s[1])
		#t,p = stats.ttest_ind(data1, data2, equal_var = False)
		text1 = 'adult %i%% interval: $%0.1f$ -- $%0.1f$ %s '%(confidence*100, q1[0],q1[1], units[var])
		text2 = 'seedling %i%% interval: $%0.1f$ -- $%0.1f$ %s '%(confidence*100, q2[0],q2[1], units[var])
		ax2.text(0.77,-0.53,'%s\n%s'%(text1,text2) , transform=ax2.transAxes, ha='right', fontsize=13)
		i +=1
		
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
	plt.savefig('%s_compare.png'%(date.today().strftime("%m%d%y")))
	#no major difference between any of the distributions (as expected)
	# we can check if there is a difference with the different climate periods using the Piekielek dataset
	wbp_adult_pie = pd.read_csv('E:\\WBP_model\\fielddata\\1950_1980_merged_data.csv')
	p_adult2 = wbp_adult_pie[wbp_adult_pie.response==1]
	#the results are radically different due to climate in 1950-1980 being substantially warmer than in the past
	
	'''
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
	'''
	
	for i in range(1,13):
		#test difference in pet and aet
		v = 'tmean'
		plt.subplot(3,4,i)
		data1 = (p_adult['tmax%i'%i]+p_adult['tmin%i'%i])/2
		data2 = (p_seed['tmax%i'%i]+p_seed['tmin%i'%i])/2
		compareDistributions(data1,data2,labels)
		plt.title('%s for month %i'%(v,i))
		plt.legend()
	t,p = stats.ttest_ind(p_adult.pet8-p_adult.aet8, p_seed.pet8-p_seed.aet8, equal_var = False)
	
	#consider the scatter matrix by group
	#change the adult presence to class 3?
	wbp_adult.response = wbp_adult.response*3
	wbp_grouped = wbp_seed.append(wbp_adult)
	wbp_sub = wbp_grouped[['response', 'tmax8', 'tmin1', 'pack5', 'pack9']]
	factor_scatter_matrix(wbp_sub[wbp_sub.response!=0], 'response')
'''