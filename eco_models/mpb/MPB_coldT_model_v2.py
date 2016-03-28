#Title: MPB_coldT_model_v2.py
#Author: Tony Chang #Adapted from Regniere and Bentz 2007 
#Date: 02.02.2015
#Abstract: This is going to be the store house of all the functions for the final cold tolerance model.
#V2 changes applied to functions to deal with nxn array of temperature

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
#Regniere defines the distribution of supercooling point (SCP) for mountain pine beetle as a logistic family probability function
#if we can define this distribution (parameterize it correctly based on the data) then we can recreate his model
#and thus sample from this distribution when we are given a specific temperature, to get the probability or a 
#SCP that defines survival (or death)

#There are 3 distributions to define as there are three defined states of SCP given the amount of 
#anti-freeze proteins built up, as well as gut clearing

####PARAMETERS########################################
#Equation 1
#three defined alphas for each of the models (which is the mean of the distribution)
alpha_1 = -9.8
alpha_2 = -21.2
alpha_3 = -32.3
#three defined betas which are the distributions variance
beta_1 = 2.26
beta_2 = 1.47
beta_3 = 2.42
alpha = np.array([alpha_1, alpha_2, alpha_3]) 
beta = np.array([beta_1, beta_2, beta_3]) 
######################
#Equation 3
rho_G = 0.311		#maximum gain rate
sigma_G = 8.716		#spread of the gain temperature response
######################
#Equation 5 
mu_G = -5.0			#optimum gain temperature at C=0
kappa_G = -39.3		#optimum gain temperature vs. C
######################
#Equation 4
rho_L = 0.791 		#maximum loss rate
sigma_L = 3.251		#spread of the loss temperature response
######################
#Equation 6
mu_L = 33.9			#optimum loss temperature at C=0
kappa_L = -32.7		#optimal loss temperature vs C
######################
#Equation 9
#Threshold C for State 1-2 transition
lambda_0 = 0.254  	#Threshold C for State 1-2 transition
lambda_1 = 0.764  	#Threshold C for State 2-3 transition
lambda_2 = 0.5 		#Threshold C for 100% State 2
lbda = np.array([lambda_0, lambda_1, lambda_2])
#######################
#phloem temperature parameter
delta_max = 3.25 #average between the north and south sides of the bole
######################################################

def leapYearCheck(year):
	if (year % 4) == 0:
		if (year % 100) == 0:
			if (year % 400) == 0:
				return(True)
			else:
				return(False)
		else:
			return(True)
	else:
		return(False)
		
def SCP(alpha, beta, T):
	#input the mean of the distribution (alpha),variance (beta) and temperature array
	#output the logistic probability distribution of super cooling point (SCP) for the given state
	X = np.exp(-(T-alpha)/beta)/(beta*(1+np.exp(-(T-alpha)/beta))**2)
	return(X)

#To define transition points in the population from State 1 to 2 to 3, the total populations degree of cold hardening is 
#defined by the continuous variable C (where C ranges from 0 to 1). Where the arbitrary C = 0.5 represents the point where 100% of the individuals
#are in State 2 of SCP. But, we still need to define the threshold of C, of when parts of the population start using
#X_2 defined as when C = lambda_0 and the threshold of C, when parts of the population start using X_3, defined as
#when C = lambda_1. 

def optTemp(mu, kappa, C):
	#For gain temperature: T_G = mu_G + (kappa_G * C) #Eq(5)
	#For loss temperature: T_L = mu_L + (kappa_L * C) #Eq(6)
	oT = mu + (kappa * C)
	return(oT)

#Regniere assumes that the degree of cold hardening C is monotonic and asymptotically approaching 1, although this is
#unfair (beetles may regress in cold tolerance under a month long warm spell), we use this assumption just to recreate
#his model.
	
def changeInColdTol(R, tau, rho, sigma, oT):
	#where tau and R are the daily mean and range of phloem temperatures (deg C)
	#G and L are defined by logistic probability distribution functions as well
	#parameters sigma_G and sigma_L are spread factors and 
	#rho_G and rho_L are the maximum rates at optimum temperatures T_G and T_L 
	#optimal temperature are modeled by 
	#For gain: G = R * rho_G * ((np.exp(-(tau-T_G)/sigma_G)/(sigma_G*(1+np.exp(-(tau-T_G)/sigma_G))**2) #Eq(3)
	#For loss: L = R * rho_L * ((np.exp(-(tau-T_L)/sigma_L)/(sigma_L*(1+np.exp(-(tau-T_L)/sigma_L))**2) #Eq(4)
	changeFactor = R * rho * ((np.exp(-(tau-oT)/sigma))/(sigma*(1+np.exp(-(tau-oT)/sigma))**2))
	return(changeFactor)

def deltaC(G,L,C,t):
	#where G is the percent gain in cold tolerance and L is the percent loss in cold tolerance. 
	#how much C changes every time step is defined by dC/dt = (1-C)*G - C*L 
	#note that Regniere limits this value to |0.2| which means that the fastest cold tolerance can be totally lost or gained is 5 days.
	#could this be different?
	dC = np.zeros(np.shape(C))
	if (t < 365): 
		if(np.any(C<0.5)):
			dC[C<0.5] = (1-C[C<0.5])*G[C<0.5]
	elif (t>= 365): 
		dC = (1-C)*G - (C*L) #Eq(2)
	elif (np.any(C>=0.5)):
		dC = (1-C[C>=0.5])*G[C>=0.5] - (C[C>=0.5]*L[C>=0.5]) #Eq(2)
	#Regnieres gain/loss limit check
	thr = 0.2
	if (np.any(np.abs(dC) > thr)):
		dC[np.abs(dC) > thr] = 0.2 * np.sign(dC[np.abs(dC) > thr])	
	return(dC)

def phloemMaxTemp(delta_max, T_max, T_min):
	#Phloem temperatures were obtained by modifying daily minimum and maximum air temperatures according to Bolstad et al 1997
	#where tau_min and tau_max are the minimum and maximum phloem temperatures in deg C
	tau_max = T_max + (delta_max *((T_max-T_min)/24.4)) #Eq(11)
	return(tau_max)
	
def phloemMinTemp(T_min):
	tau_min = T_min + 1.8	#Eq(12)
	return(tau_min)

def popProp(C, lbda):
	dif_lambda1 = lbda[2] - lbda[0]  
	dif_lambda2 = lbda[1] - lbda[2]
	popFac = np.zeros((3,np.shape(C)[0],np.shape(C)[1]))
	first_case = (C<=lbda[0])
	second_case = np.logical_and(~first_case,(C<lbda[2]))
	third_case = np.logical_and(~first_case,np.logical_and(~(C<lbda[2]), (C<lbda[1])))
	fourth_case = (C>=lbda[1])
	if (np.any(first_case)):
		popFac[0][first_case] = 1 	#first case, 100% in stage 1
	if (np.any(second_case)): 		#second case, stage 1-2 transition
		popFac[1][second_case] = (C[second_case]-lbda[0])/dif_lambda1
		popFac[0][second_case] = 1-popFac[1][second_case]
	if (np.any(third_case)):		#third case, stage 2-3 transition
		popFac[2][third_case] = (C[third_case]-lbda[2])/dif_lambda2
		popFac[1][third_case] = 1-popFac[2][third_case]
	if (np.any(fourth_case)): 		#C >=lambda_1
		popFac[2][fourth_case] = 1
	return(popFac) #returns a 3 x n x m population fraction array
	
def medianSCP(p, alpha):
#returns the median SCP value for the population
	lt50 = alpha[:,None,None]*p #Make alpha a 3D array to multiply against p
	return(np.sum(lt50, axis=0))

def survive(p, T, alpha, beta):
#solves for the survival function at current time step
	alpha_array = alpha[:,None,None]*np.ones(np.shape(p))
	beta_array = beta[:,None,None]*np.ones(np.shape(p))
	s = np.sum((p/(1+np.exp(-(T - alpha_array)/beta_array))),axis=0)
	return(s)

def surviveCheck(s_t1,s_t):
	return(np.minimum(s_t1,s_t))
	
def plotPop(C, T=None, lbda = lbda, alpha=alpha, beta=beta, save = 'n', name=None):
#plots the distribution within each stage given the C
	if T == None: #if the user does not specify the T range
		T = np.linspace(-50,5,250)
	scpDist = []
	for i in range(len(alpha)):
		scpDist.append(SCP(alpha[i],beta[i], T))
	scpDist = np.array(scpDist)
	popFac = popProp(C, lbda)
	props = []
	for i in range(len(popFac)):
		plt.fill(T, popFac[i]*scpDist[i], label ='State %s'%(i+1), alpha = 0.5)
		props.append(popFac[i]*scpDist[i])
	max_p = np.max(np.array(props))
	lt50 = medianSCP(popFac, alpha)
	ax = plt.subplot(111)
	plt.vlines(lt50, 0, max_p, color = 'red', linestyle = '--', label = 'Median SCP')
	plt.xlabel('Supercooling Point (SCP)')
	plt.ylabel('Simulated Normalize Frequency')
	plt.title('Supercooling state C: %0.2f; Median SCP: %0.2f$^oC$'%(C, lt50))
	plt.grid()
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	if save =='y':
		plt.savefig('dist_plot%s.png'%(n))
	plt.show()
	return()

def runModel(Tmin, Tmax, year = datetime.date.today().year, C_p=None):
#model requires daily Tmin and Tmax air temperatures
#data should begin at Aug 1 and end at Aug 1 of the following year
#default year is the current year, but user can enter it
#if the C is entered, then the model will initialize with the given C and s must be provided
	#Tmean = np.mean([Tmin,Tmax], axis=0) 
	Trange = Tmax-Tmin
	phTmax = phloemMaxTemp(delta_max, Tmax, Tmin)
	phTmin = phloemMinTemp(Tmin)
	phTmean = np.mean([phTmin,phTmax], axis=0)
	phTrange = (phTmax-phTmin)
	#initialize arrays here for storage
	#now we need to generate a function that calculates the change in C per time step (daily)
	t_days = len(phTmean) #find how many values
	#geneate some arrays for storage
	C = []
	P1 = []
	P2 = []
	P3 = []
	s = []
	lt50 = []
	date = []
	if leapYearCheck(year):
		julien_convert = 214
	else:
		julien_convert = 213	
	for t in range(t_days):
	#first every time step we need to calculate the L and G values which are a function of tau and R
		julien_date = t + julien_convert
		tau = phTmean[t]
		R = phTrange[t]
		if (t == 0):
			if (C_p == None):
			#for the first day C_t should be 0 and survival probability should be 1
				C_t = np.zeros(np.shape(Tmin[0]))
				otG = np.ones(np.shape(Tmin[0]))* mu_G
				otL = np.ones(np.shape(Tmin[0]))* mu_L
				new_s = np.ones(np.shape(Tmin[0]))
			else:	#if there is a value in C_t
				C_t = C_p
				otG = optTemp(mu_G, kappa_G, C_t) #gain and loss are defined by the same function, but have differing parameters
				otL = optTemp(mu_L, kappa_L, C_t)
				new_s = np.ones(np.shape(Tmin[0]))
				# we want to assume that the population regenerates at the end of the year. So the survival is back to one
			#but the cold hardening status is dependent on the previous year?
		else:
			delC = deltaC(G,L,C_t,julien_date)
			C_t = C_t + delC
			otG = optTemp(mu_G, kappa_G, C_t) #gain and loss are defined by the same function, but have differing parameters
			otL = optTemp(mu_L, kappa_L, C_t)
			s_t = survive(P_t, phTmin[t], alpha, beta)
			new_s = surviveCheck(s[t-1],s_t)		
		L = changeInColdTol(R, tau, rho_L, sigma_L, otL)
		G = changeInColdTol(R, tau, rho_G, sigma_G, otG) 
		P_t = popProp(C_t, lbda)
		s.append(new_s)
		lt50.append(medianSCP(P_t, alpha))
		C.append(C_t)
		P1.append(P_t[0])
		P2.append(P_t[1])
		P3.append(P_t[2])
		date.append(julien_date)
	P1 = np.array(P1)
	P2 = np.array(P2)
	P3 = np.array(P3)
	s = np.array(s)
	C = np.array(C)
	tau = phTmean
	R = phTrange
	
	lt50 = np.array(lt50)
	date = pd.date_range('8/1/%s'%year, periods=t_days)
	#change the output here? perhaps create an option to have differing outputs
	#out = np.array([C,s])
	out = np.array([tau, R, phTmin, lt50, s, C, P1, P2, P3])
	#data = {'Date': date, 'Phm_Tmin' : phTmin, 'tau' : phTmean, 'R' : phTrange,'P1' : P[:,0], 'P2': P[:,1], 'P3': P[:,2], 'C' : C, 'ProbS': s, 'LT50': lt50}
	#labels = ['Date', 'Phm_Tmin', 'tau', 'R', 'P1', 'P2', 'P3', 'C', 'ProbS', 'LT50']
	#out = pd.DataFrame(data, columns=labels)
	#out = np.array([date, phTmin, phTmean, P, C, s, lt50],dtype=[(labels[0], 'pd.tslib.Timestamp'),(labels[1],'f8'),(labels[2],'f8'),(labels[3],'f8'),(labels[4],'f8'),(labels[5],'f8'),(labels[6]),'f8']) #fix the output to pandas (regional mean) or numpy array
	return(out)	

