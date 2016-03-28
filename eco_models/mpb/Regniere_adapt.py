import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Regniere defines the distribution of supercooling point (SCP) for mountain pine beetle as a logistic family probability function
#if we can define this distribution (parameterize it correctly based on the data) then we can recreate his model
#and thus sample from this distribution when we are given a specific temperature, to get the probability or a 
#SCP that defines survival (or death)

#There are 3 distributions to define as there are three defined states of SCP given the amount of 
#anti-freeze proteins built up, as well as gut clearing

alpha = np.array([]) #three defined alphas for each of the models (which is the mean of the distribution)
beta = np.array([]) #three defined betas which are the distributions variance
T = np.array([]) #the temperature from field data

X = np.exp(-(T-alpha)/beta)/(beta*(1+np.exp(-(T-alpha)/beta))**2))

#To define transition points in the population from State 1 to 2 to 3, the total populations degree of cold hardening is 
#defined by the continuous variable C (where C ranges from 0 to 1). Where the arbitrary C = 0.5 represents the point where 100% of the individuals
#are in State 2 of SCP. But, we still need to define the threshold of C, of when parts of the population start using
#X_2 defined as when C = lambda_0 and the threshold of C, when parts of the population start using X_3, defined as
#when C = lambda_1. 

#Regniere assumes that the degree of cold hardening C is monotonic and asymptotically approaching 1, although this is
#unfair (beetles may regress in cold tolerance under a month long warm spell), we use this assumption just to recreate
#his model.

#how much C changes every time step is defined by dC/dt = (1-C)*G - C*L 
deltaC = (1-C)*G - C*L
#where G is the percent gain in cold tolerance and L is the percent loss in cold tolerance. 

#G and L are defined by logistic probability distribution functions as well

G = R * rho_G * ((np.exp(-(tau-T_G)/sigma_G)/(sigma_G*(1+np.exp(-(tau-T_G)/sigma_G))**2)

L = R * rho_L * ((np.exp(-(tau-T_L)/sigma_L)/(sigma_L*(1+np.exp(-(tau-T_L)/sigma_L))**2)

#where tau and R are the daily mean and range of phloem temperatures (deg C)
#this will later be estimated with the allometric relationship of whitebark pine phloem to air temperature 
#^^^ THIS NEEDS TO BE LOOKED UP!!
#parameters sigma_G and sigma_L are spread factors and 
#rho_G and rho_L are the maximum rates at optimum temperatures T_G and T_L 
#optimal temperature are modeled by 

T_G = mu_G + (kappa_G * C)
T_L = mu_L + (kappa_L * C)

#Regniere also assumes that it is highly unlikely that once C>0.5 (100% of population has entered State 2),
#the season has reached a point where temperatures can return to summer SCP conditions. So beetles can not regress
#so we can redefine deltaC as
t_end = 365 #last day of year
t_begin = 213 #beginning of SCP progression (could be 214 if leap year)
t = #current day
t_range = np.arange(t_begin, t)

if ((t<t_end) and (C<0.5)):
	C_t1 = np.sum(1-C_t)*G_t
else if ((t>=t_end) or (C>=0.5)):
	C_t1 = np.sum((1-C_t)*G_t - (C_t *L_t))
#this needs work



