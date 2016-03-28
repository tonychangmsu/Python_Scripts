#Title: 	mpb_phen_model.py
#Author: 	Tony Chang
#			(Adapted from Powell and Logan 1999)
#Date: 		06.17.2015
#Abstract:	Development rate model for mountain pine beetle that is dependent on an initialization period (start date)
#			and hourly temperatures

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import powell_logan as mpb
import scipy.stats as sp
#%load_ext autoreload
#%autoreload 2
######################### MAIN ########################

workspace = "E:\\MPB_model\\mpb_phenology\\data\\"
temp_filename = '%s%s' %(workspace, 'temp_06172015.txt') 
param_filename = '%s%s' %(workspace, 'p_new_06172015.txt') 
t = pd.read_csv(temp_filename, header = 0)
p = pd.read_csv(param_filename, delimiter =',')
t_array = t.Temperature.values
p_array = p.values
t_hm = mpb.hourlyTemp(t_array)
#lets consider only half of the dataset because we don't need all 4 years.
t_hm_half = t_hm[:(365*2),:]
#t_hm_half = t_hm[:365,:]

#dev_day, ndays = mpb.devPerDay(t_array, p_array)
dev_day = mpb.devDays(t_hm, p_array)

#the time to run this variation is not too bad. consider this method for a daily rate...

#now we have each of the development rates for each day we can look into the trap_devrats_new function
#need to solve the median emergence day for a starting date of 1 through 365 (or 366 for leap year)

devcomp = np.zeros((365,8)) #array to save the development day
for start_date in range(365):
#so we need to just find out when we get to a one for the first stage and progress this way for all the lifestages
	for i in range(8):
		if i == 0: #for egg state
			checker = np.max(np.cumsum(dev_day[i][start_date:]))
			if checker<=1:
				break
			else:
				dev_date = np.where(np.cumsum(dev_day[i][start_date:])>=1)[0][0]
				devcomp[start_date][i] = dev_date + start_date
		else:
			checker = np.max(np.cumsum(dev_day[i][start_date:]))
			if checker<=1:
				break
			else:
				dev_date = np.where(np.cumsum(dev_day[i][devcomp[start_date][i-1]:])>=1)[0][0]
				devcomp[start_date][i] = dev_date + devcomp[start_date][i-1]

#now we have the total development days for all egg laying dates. 
#the question is what metric we are trying to measure
#it may be beneficial to select a specific start date? 
#or initialize for 4 years and see if they all converge in the end...
#seems like October 29 is the day of final emergence for almost all simulations

g = devcomp[:,-1]
g[g>365] = g[g>365] -365 #so we have to fix some issues, I need to figure out how to get things initialized for a single year first at all 
#start dates. Then we record the emergence date after that initial year (maybe two). 
#what we can do is take the second year level of development and use that to initialize there?

#then we need to define adaptive emergence risk as the ratio of beetles that emerged in a the year divided by 365 (all possible days). So 
#a high emergence risk would be indicated with 90% of the days having beetles emerge that year. 
#but if we continue with this exercise over time, then we should get more? because we will see synchrony over time?

count = plt.hist(g, bins = 365) #this here makes sense? couldn't we just do a count in each bin?
probs = count[0]/365*100
#okay, so now we can get a probability distribution for this function regarding the emergence date
#so we should select the date where the probability is greater than 50% (dependent on the number of bins)
#we can solve for the distribution 
#perhaps just use the mode?
out = np.max(count[0])
prob_out = out/365		

#develop a new module that generates hourly temperatures for TopoWX
#cosine option may be the best as the SAWTOOTH model is very crude.