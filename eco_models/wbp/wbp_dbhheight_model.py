#Title: DBH to height relationship models for Pinus
#(From Huag and Titus 1992 Comparison of nonlinear height diameter functions for major Alberta tree species)
# model for p.albicaulis, p.flexis, p.contorta
#Author: Tony Chang
import numpy as np
from matplotlib import pyplot as plt
	
def curtis_prodan(D):
# D == dbh in cm
# returns height in m
	a =1.4431
	b =0.6806
	c =0.0233
	return(1.3+ (D**2)/(a + (b*D) + (c*D**2)))

def weibull_type(D):
	a = 29.0401
	b = 0.0318
	c = 1.0902
	return(1.3 + a*(1-np.e**(-b*(D**c))))

def chapman_richards(D):
	a = 29.4214
	b = 0.0457
	c = 1.1381
	return(1.3 + a*((1-np.e**(-b*(D)))**c))

#using 8" DBH (20.32cm) as the baseline for our mature age class (in terms of height)
#should use 50-75 cm DBH to get 70 to 100 m height 
#mature speciman are between 5m and 20m Wilson 2007 Status of WBP ESRD

D = np.linspace(0, 100, 100)
plt.plot(D,curtis_prodan(D), label ='Curtis-Prodan')
plt.plot(D,weibull_type(D), label ='Weibull type')
plt.plot(D,chapman_richards(D), label = 'Chapman-Richards') 
plt.plot(D, np.ones(len(D))*33, ls ='--', color ='grey')
plt.legend(loc = 'lower right')
plt.grid()
plt.xlabel('DBH cm')
plt.ylabel('Height m')

