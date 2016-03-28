import numpy as np

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
#maximum gain rate
rho_G = 0.311
#spread of the gain temperature response
sigma_G = 8.716
######################
#Equation 5 
#optimum gain temperature at C=0
mu_G = -5.0
#optimum gain temperature vs. C
kappa_G = -39.3
######################
#Equation 4
#maximum loss rate
rho_L = 0.791 
#spread of the loss temperature response
sigma_L = 3.251
######################
#Equation 6
#optimum loss temperature at C=0
mu_L = 33.9
#optimal loss temperature vs C
kappa_L = -32.7
######################
#Equation 9
#Threshold C for State 1-2 transition
lambda_0 = 0.254
#Threshold C for State 2-3 transition
lambda_1 = 0.764
#Threshold C for 100% State 2
lambda_2 = 0.5
#######################
#phloem temperature paramter
delta_max = 3.25 #average between the north and south sides of the bole
######################################################

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
	if (t < 365) and (C<0.5):
		dC = (1-C)*G
	elif (t>= 365) or (C>=0.5):
		dC = (1-C)*G - (C*L) #Eq(2)
	#Regnieres gain/loss limit check
	thr = 0.2
	if (np.abs(dC) > thr):
		dC = 0.2 * np.sign(dC)	
	return(dC)


#this will later be estimated with the allometric relationship of whitebark pine phloem to air temperature

def phloemMaxTemp(delta_max, T_max, T_min):
	#Phloem temperatures were obtained by modifying daily minimum and maximum air temperatures according to Bolstad et al 1997
	#where tau_min and tau_max are the minimum and maximum phloem temperatures in deg C
	tau_max = T_max + (delta_max *((T_max-T_min)/24.4)) #Eq(11)
	return(tau_max)
	
def phloemMinTemp(T_min):
	tau_min = T_min + 1.8	#Eq(12)
	return(tau_min)
	
#Regniere also assumes that it is highly unlikely that once C>0.5 (100% of population has entered State 2),
#the season has reached a point where temperatures can return to summer SCP conditions. So beetles can not regress
#so we can redefine deltaC as
'''
t_end = 365 #last day of year
t_begin = 213 #beginning of SCP progression (could be 214 if leap year)
t = #current day
t_range = np.arange(t_begin, t)
C_t0 = 0 #initial C should be 0 at the beginning of the year

if ((t<t_end) and (C<0.5)):
	C_t1 = np.sum(1-C_t)*G_t
else if ((t>=t_end) or (C>=0.5)):
	C_t1 = np.sum((1-C_t)*G_t - (C_t *L_t))
'''
#this needs work
'''
#where the proportions p are #Eq(9)
C=0
p_1 = np.max(np.array([0,np.min((0.5-C)/(lambda_0-C))]))
p_3 = np.max(np.array([0,np.min((C-0.5)/(C-lambda_1))]))
p_2 = 1 - (p_1 + p_3)

p = np.array([p_1, p_2, p_3]) #population array

#The median SCP of the population is a linear combination of the median of each distribution multiplied by the 
#proportion p_i of the population in each State i:
lt50 = np.sum(p*alpha) #Eq(8)

#For any given minimum daily temperature T_t, on day t, survival is teh lowest of the previous day's survival
#and probability of survival in all three cold-hardening states, weighted by the proportion of the population
#in each state:

p_survival_t = np.min(np.array([p_survival_t_1 , np.sum(p/(1+np.exp**(-(T_t-alpha)/beta)))])) #Eq(10)
'''



