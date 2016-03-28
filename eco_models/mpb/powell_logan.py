#Title: 	powell_logan.py
#Author: 	Tony Chang
#Date: 		06.17.2015
#Abstract:	Functions for the development rate model

import numpy as np
import pandas as pd
#################### Life stage development rate models ####################

def blogan(t,p):
	'''
	function [y] = blogan(t,p)
	#	this function takes as input the vector 'temps' of temperatures and returns
	#	a vector y of the rate/temperature function 'alogan' value for each temperature in 
	#	temps, for parameters p1=psi, p2=increase rate, p3=max temperature, p4=del t 
	#	(high temp boundary)
	use:
	y=alogan(x,p)
	'''
	z1 = 1/(1+p[1]*np.exp(-p[2]*t))
	z2 = np.exp(-(p[3]-t)/p[4])
	z3 = p[0]*(z1-z2)
	#z4=z3[np.where(z3>0)] 	# original matlab code
	r = np.where(z3>0,z3,0)	# filter for rates less than zero
	return(r)

def typeiii(t,p):
	xt = t-p[4]
	tau = (p[1]-xt)/p[2]
	x1 = p[0]*(xt**2)/((xt**2)+(p[3]**2))
	x2 = p[0]*(1-np.exp(-1*tau)) 
	x3 = x1+x2-p[0]
	#z=np.reshape(np.where(x3>0, 1, 0), np.shape(x3))
	z = np.where(x3>0, 1, 0)
	q = np.where(t>p[4],1,0) #check for T less than base Temperature
	r = z*q*x3
	r = np.where(r>0, r, 0)
	return(r)

def linear_rate(t,p):
	r = p[1]*(t-p[0])
	r[np.where(r<0)] = 0 	#change all y values less than zero to 0
	return(r)

def strat(t,p):
	# Stinners developmental rate curve
	t = np.where(t>p[3],(2*p[3]-t),t)
	r = p[0]/(1 + np.exp(p[1] + p[2] * t))
	v = np.where(r>0,r,0)
	return(v)

def gallery_ln(t,p):
	'''
	function [y] = gall_ln(tmps,p)
	#	gallery length constructed as a function of temperature for mpb - see Logan et al. IUFRO 
	#	Hawaii Proceedings
	#	p(1) - P(3) are est parameters; p(4) is tau; p(5) is base Temperature
	'''
	t = t-p[4]
	t = np.where(t>0, t, 0)
	tau = t/p[3]
	r = ((p[0] * (np.exp(p[1]*t**p[2]) - np.exp(-1*tau))) * 2.54)/32. # the 2.54 converts from in to cm
	#divide by 32 to normalize to median
	r = np.where(r>0, r, 0)
	return(r)

############################################################################

def hourlyTemp(t):
	#Input: 	cuts the hourly list of temperatures to single daily arrays (numpy array)
	#Output: 	return matrix of 25 hour daily temperatures for dailyDev function processing
	#			Note that the last day will be discarded because there is not a 25 hour to reference
	ndays = int(len(t)/24) #find the number of days of date from the temperature data set
	y_raw = np.reshape(t,(ndays,24)) # ndays X 24 hrs matrix of temperatures
	a = np.hstack((y_raw[1:,0],0)) #additional column to be added for endpoint with 0 for the last value
	yy_raw = np.vstack((y_raw.T,a)).T #Add end point for integration
	yval = np.delete(yy_raw,-1,0) # eliminate last day (row) with no end point
	return(yval)
	
def dailyDev(t,p,lstage):
	#Input: a temperature array (t) of 25 hourly temperature values
	#		a parameter array for the various life stages
	#Note:	Not necessary to use the rotated output from the life stage calculations, 
	#		just calculate np.trapz from a different axis @t.chang 10.27.2015
	if lstage ==0:
		return(np.trapz((blogan(t-5,p[lstage,:])/24).T, axis=0)) #egg #finds the amount of development that has occurred at each day given temperature
	elif lstage ==1:
		return(np.trapz((blogan(t-5,p[lstage,:])/24).T, axis=0)) #L1
	elif lstage ==2:
		return(np.trapz((blogan(t-10,p[lstage,:])/24).T, axis=0)) #L2
	elif lstage ==3:
		return(np.trapz((typeiii(t,p[lstage,:])/24).T, axis=0)) #L3
	elif lstage ==4:
		return(np.trapz((linear_rate(t,p[lstage,:])/24).T, axis=0)) #L4
	elif lstage ==5:
		return(np.trapz((linear_rate(t,p[lstage,:])/24).T, axis=0)) #pupae
	elif lstage ==6:
		return(np.trapz((strat(t,p[lstage,:])/24).T, axis=0)) #Pre-ovipositional adult
	elif lstage ==7:
		return(np.trapz((gallery_ln(t,p[lstage,:])/24).T, axis=0)) # Ovipositional adult insane bug in python?! need to assign this first
	else:
		return(print("Error, input a valid life stage"))
		
def devDays(t_hm, p=None):
	#Input:		Inputs the daily hourly temperature matrix, parameter values
	#Output:	Calculates development rates with an iterative method
	if p==None: #if parameters not entered use Default
		workspace = "E:\\MPB_model\\mpb_phenology\\data\\"
		param_filename = '%s%s' %(workspace, 'p_new_06172015.txt') 
		p = pd.read_csv(param_filename, delimiter =',').values #parameters for each development rate model 
	total_lstages = 8
	d_d = np.zeros((total_lstages,len(t_hm)))
	for lstage in range(total_lstages):
		for i in range(len(t_hm)):
			d_d[lstage][i] = dailyDev(t_hm[i], p, lstage)
	return(d_d)
	
#for matrices calculations....

def dailyDevArray(t,p,lstage):
	#Input: a temperature matrix (t) of 25 hourly temperature values
	#		a parameter array for the various life stages
	if lstage == 0:
		return(np.trapz((blogan(t-5,p[lstage,:])/24), axis=0)) #egg #finds the amount of development that has occurred at each day given temperature
	elif lstage ==1:
		return(np.trapz((blogan(t-5,p[lstage,:])/24), axis=0)) #L1
	elif lstage ==2:
		return(np.trapz((blogan(t-10,p[lstage,:])/24), axis=0)) #L2
	elif lstage ==3:
		return(np.trapz((typeiii(t,p[lstage,:])/24), axis=0)) #L3
	elif lstage ==4:
		return(np.trapz((linear_rate(t,p[lstage,:])/24), axis=0)) #L4
	elif lstage ==5:
		return(np.trapz((linear_rate(t,p[lstage,:])/24), axis=0)) #pupae
	elif lstage ==6:
		return(np.trapz((strat(t,p[lstage,:])/24), axis=0)) #Pre-ovipositional adult
	elif lstage ==7:
		return(np.trapz((gallery_ln(t,p[lstage,:])/24), axis=0)) # Ovipositional adult insane bug in python?! need to assign this first
	else:
		return(print("Error, input a valid life stage"))

def devArray(t,p,lstage):
	#Input: a temperature matrix 
	#		a parameter array for the various life stages
	#Output: the rate of development for the requested life stage per day
	#		 since equations provide rates at as % development per day @changed to maintain dev per day (@t.chang 2015.11.03)
	if lstage == 0:
		return(blogan(t-5,p[lstage,:])) #egg #finds the amount of development that has occurred at each day given temperature
	elif lstage ==1:
		return(blogan(t-5,p[lstage,:])) #L1
	elif lstage ==2:
		return(blogan(t-10,p[lstage,:])) #L2
	elif lstage ==3:
		return(typeiii(t,p[lstage,:])) #L3
	elif lstage ==4:
		return(linear_rate(t,p[lstage,:])) #L4
	elif lstage ==5:
		return(linear_rate(t,p[lstage,:])) #pupae
	elif lstage ==6:
		return(strat(t,p[lstage,:])) #Pre-ovipositional adult
	elif lstage ==7:
		return(gallery_ln(t,p[lstage,:])) # Ovipositional adult insane bug in python?! need to assign this first
	else:
		return(print("Error, input a valid life stage"))

		
def devDaysArray(t_hm, p=None):
	#Input: 	Inputs the daily hourly temperature in integer format (*100) matrix and parameter values
	#Output: 	Calculates development rates with an iterative method
	#Note:		A non-iterative method would be faster
	if p==None: #if parameters not entered use Default
		workspace = "E:\\MPB_model\\mpb_phenology\\data\\"
		param_filename = '%s%s' %(workspace, 'p_new_06172015.txt') 
		p = pd.read_csv(param_filename, delimiter =',').values #parameters for each development rate model 
	total_lstages = 8 
	shp = np.shape(t_hm) #find the dimensions of t_hm
	d_d = np.zeros((total_lstages, shp[0], shp[1], shp[2])) #now we have a array to store the t_hm for each life stage
	#this is too large a matrix to store all the life stages, so instead we will return each one as a separate variable
	#d_d = []
	for lstage in range(total_lstages):
		#temp = np.zeros(shp)
		for i in range(shp[0]):
			#temp[i] = dailyDevArray(t_hm[i]/100, p, lstage) #each single calculation array will cost 16 GB!
			#need to find a better solution such as using float32 rather than float64
			#we can still do this, but require writing each output for each life stage, at each hour
			#this might work and then we just calculate based on that, since we only have to consider a few year
			#chunks at a time
			#d_d.append(temp) #store in a list?
			d_d[lstage][i] = dailyDevArray(t_hm[i]/100, p, lstage)
	return(d_d)
	
############################ OLD FUNCTIONS ############################
def devPerDay(t,p):
	'''
	Abstract
	program to compute the development per day for the trap_devrats_new model
	#	for temperatures in x - this is a helper function for trap_devrats_new
	#	note: one day will be lost due to requiring an end point for integration.  
	#	resulting in loosing one day of temperatures.   Therefore,
	#	you will need to add one day of temperatures to the file, ie. for one year you need
	#	8760+24 hourly temperatures; for 2 years of data, 2*8760+24 hourly temperatures,
	#	ect.
	usage:
	#	dev_day,ndays = dev_per_day(x,p)
		
	input
	#	x=row vector of hourly temperatures, this needs to be an even multiple of 24!
	#	p - array of developmental parameters for the trap_devrats_new model
	#	on output:
	#	dev_day - the development per day for an observed temperature cycle
	#	ndays - the number of days in x
	Known Bugs - 
	#	see also trap_devrats_new 
	'''
	ndays = int(len(t)/24) #find the number of days of date from the temperature data set
	y_raw = np.reshape(t,(ndays,24)) # ndays X 24 hrs matrix of temperatures
	a = np.hstack((y_raw[1:,0],0)) #additional column to be added for endpoint with 0 for the last value
	yy_raw = np.vstack((y_raw.T,a)).T #Add end point for integration
	yval = np.delete(yy_raw,-1,0) # eliminate last day (row) with no end point
	#ndays=ndays-1 # set new size to number of days, for this case that should be 1459 days
	#dev_day = np.zeros((7,ndays))
	dev_day = []
	# produces an 8 no. life stages (rows) X ndays (columns) of daily developmental indices

	dev_day.append(np.trapz((blogan(yval-5,p[0,:])/24).T, axis=0)) #egg #finds the amount of development that has occurred at each day given temperature
	dev_day.append(np.trapz((blogan(yval-5,p[1,:])/24).T, axis=0)) #L1
	dev_day.append(np.trapz((blogan(yval-10,p[2,:])/24).T, axis=0)) #L2
	dev_day.append(np.trapz((typeiii(yval,p[3,:])/24).T, axis=0)) #L3
	dev_day.append(np.trapz((linear_rate(yval,p[4,:])/24).T, axis=0)) #L4
	dev_day.append(np.trapz((linear_rate(yval,p[5,:])/24).T, axis=0)) #pupae
	dev_day.append(np.trapz((strat(yval,p[6,:])/24).T, axis=0)) #Pre-ovipositional adult
	dev_day.append(np.trapz((gallery_ln(yval,p[7,:])/24).T, axis=0)) # Ovipositional adult insane bug in python?! need to assign this first
	dev_day = np.array(dev_day)
	
	return(dev_day, ndays) #output a percent development per day by life stage...
	
def trap_devrats_new(dev_day,ndays,da_start,nyrs = 1):
	'''
	%***************************************************************************
	%* NOTE: ARGUMENT m_g_t HAS BEEN COMMENTED OUT FOR CALL from G_curve       *
	%* mean_generation_time] = trap_devrats_new(p,dev_day,ndays,da_start,nyrs);*
	%***************************************************************************
	%
	% Abstract;  program to compute and solve for median days for completing the eight
	%   life stages MPB Model 
	%
	% usage:
	%   [med_day_emerg,phase_space,years_per_generation,mean_generation_time] ...
	%       = trap_devrats_new(dev_day,ndays,da_start,nyrs);
	%	
	%   input
	%       dev_day - the development per day for an observed temperature cycle
	%       ndays - number of days to run the model to get emergence - emergence date will be 
	%           reduced modulo 365 - generally this is the ndays returned by dev_per_day
	%       da_start - the julian day for initial oviposition
	%       nyrs - the number of simulation years - this is actually generations! 
	%trap_devrats_new(
	%   Output:
	%       Med_day_emergence - this is an array of size (nyrs,8) that contains julian emergence
	%           dates for the 8 life stages
	%       phase_space - the x,y phase space for median adult emergence(n)/oviposition(n+1) - size
	%           (nyr,2)
	%       years_per_generation - the number of years per generation (nyrs,1)
	%       mean_generation_time - The mean generation time for each of nyrs (nyr,1) - computation
	%           of this variable will result in a catastrophic error if nyrs=1!
	%
	%   Known Bugs - This function has not been rigorously tested for very fast (<1 yr) or very slow
	%       (>>1 yr) generation times
	%
	%   see also dev_per_day
	%
	% to get actual emergence date from any initial condition: (1) set
	% nyrs=1;  (2) comment out the modulo loop (3) set p_s=M_D_E(i,7)
	% (4) save x=tmp(:,3);
	%
	% compute the proportion of the life stage completed for each day in the temperature data set for each life stage in the model
	% simulation loop starts here
	'''
	#da start is unknown. Each time we calculate the emergence date, we have to initialize for each of the possible oviposition start
	#so perhaps we need to determine the emergence date for all 365 oviposition starts and then look at the distribution of the emergences
	
	#med_day_emerg= np.ones((nyrs,8))*np.nan
	med_day_emerg = np.zeros((nyrs,8+1)) #need an initial year for full development?
	years_per_generation = np.zeros((nyrs,1))
	phase_space = np.zeros((nyrs,2))
	dd = np.zeros((np.shape(dev_day)[0]+1,np.shape(dev_day)[1]+1)) #hack to get the indexing to match that of original matlab code...
	dd[1:,1:] = dev_day
	dev_day = dd
	for i in range(1,nyrs+1):
		test=np.cumsum(dev_day[1,da_start:],axis=0) #test if the first life stage reaches maturity (which it should)
		if(np.max(test)<1.0): #if there are no values in test that are greater than 1 exit the loop
			break
		med_day_emerg[i,1] = fnd1idx(test,1,1)+da_start #initialize the first day stage median date of emergence
		#need to add 1 to da_start and fnd1idx to account for indexing at 0 
		for j in range(2,9): #else from 1 to 7 the different development stages
			test=np.cumsum(dev_day[j,int(med_day_emerg[i,j-1]):],axis=0) #find the index where the first cumsum is greater than 1
			if(np.max(test)<1.0):
				break
			med_day_emerg[i,j] =(med_day_emerg[i,j-1]+fnd1idx(test,1,1)) #initialize the first day stage median date of emergence
		#if(len(test) ==0 or np.max(test)<1.0):
		#	break
		#set beginning phase space 
		phase_space[i-1,0]=da_start
		years_per_generation[i-1,0]=(med_day_emerg[i,8]-da_start)/365 #med day emerg is off by one because indexing starts at 0, but actual years are by 1
		da_start=np.mod(med_day_emerg[i,8],365) # modulo reduction
		phase_space[i-1,1]=np.mod(med_day_emerg[i,8],365)
		if (da_start==0):
			da_start=1
		phase_space[i-1,1]=np.mod(med_day_emerg[i,7],365)
	return(med_day_emerg[1:,1:],phase_space,years_per_generation)
	