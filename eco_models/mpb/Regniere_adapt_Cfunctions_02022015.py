import numpy as np
import Regniere_adapt_01292015 as mpb
import matplotlib.pyplot as plt
import pandas as pd

#so let's build the function that sets the populations
#first off we define the functionm

def popProp(C, lambda_0, lambda_1, lambda_2):
	dif_lambda1 = lambda_2 - lambda_0  
	dif_lambda2 = lambda_1 - lambda_2
	popFac = np.zeros(3)
	if (C<=lambda_0): #first case, 100% in stage 1
		popFac[0] = 1
	elif (C<lambda_2): #second case, stage 1-2 transition
		popFac[1] = (C-lambda_0)/dif_lambda1
		popFac[0] = 1-popFac[1]
	elif (C<lambda_1):
		popFac[2] = (C-lambda_2)/dif_lambda2
		popFac[1] = 1-popFac[2]
	else: #C >=lambda_1
		popFac[2] = 1
	return(popFac)
	
def plotPop(popFac, scpDist, C, T, alpha, n=0):
	props = []
	for i in range(len(popFac)):
		plt.fill(T, popFac[i]*scpDist[i], label ='State %s'%(i+1), alpha = 0.5)
		props.append(popFac[i]*scpDist[i])
	max_p = np.max(np.array(props))
	lt50 = medianSCP(popFac, alpha)
	plt.vlines(lt50, 0, max_p, color = 'red', linestyle = '--', label = 'Median SCP')
	plt.xlabel('Supercooling Point (SCP)')
	plt.ylabel('Simulated Normalize Frequency')
	plt.title('Supercooling state C: %0.2f; Median SCP: %0.2f$^oC$'%(C, lt50))
	plt.legend()
	plt.grid()
	plt.savefig('dist_plot%s.png'%(n))
	plt.show()
	return()

####MAIN####
testT = np.linspace(-50,5,200)
scpState1 = mpb.SCP(mpb.alpha[0],mpb.beta[0], testT)
scpState2 = mpb.SCP(mpb.alpha[1],mpb.beta[1], testT)
scpState3 = mpb.SCP(mpb.alpha[2],mpb.beta[2], testT)
scpDist = np.array([scpState1, scpState2, scpState3])

#looks like the population distribution works. We might need to determine the median population cold tolerance. 
#but remembering that this value does not mean all the beetles will be dead. Perhaps explore the survival function?
#First thing to focus on is C over time

#####DATA#############################################
#defined in Table 2 of Regniere and Bentz 2007
#get the temperature data from MPB model
datafile = 'E:\\MPB_model\\MPB_phenology\\DATA\\temp.txt'
raw_temp = pd.read_csv(datafile, header = None)
T = raw_temp.dropna() #the temperature from field data with the na removed
#note that this data set represents 4 years of data hourly 
#get the daily highs and lows
Tmin = []
Tmax = []
Tmean = []

for i in range(0, len(T), 24):
	Tmin.append(np.min(T[i:i+24])[0])
	Tmax.append(np.max(T[i:i+24])[0])
	Tmean.append(np.mean(T[i:i+24])[0])
Tmin = np.array(Tmin)
Tmax = np.array(Tmax)
Tmean = np.array(Tmean)
Trange = Tmax-Tmin

phTmax = mpb.phloemMaxTemp(mpb.delta_max, Tmax, Tmin)
phTmin = mpb.phloemMinTemp(Tmin)
phTmean = (phTmax+phTmin)/2
phTrange = (phTmax-phTmin)
#we may only want the days of Aug 1 to next Aug 1 (day 213 + 365)
aphTmin = phTmin[212:212+365]
aphTmax = phTmax[212:212+365]
aphTmean = phTmean[212:212+365]
aphRange = phTrange[212:212+365]

######################################################
#assign C here
C_0 = 0 #C always starts at zero for the julien date of 213 or August 1
#now we need to generate a function that calculates the change in C per time step (daily)
t_days = len(aphTmean) #find how many values
C_t = C_0
C_array = [C_t]
P = [popProp(C_t, mpb.lambda_0, mpb.lambda_1, mpb.lambda_2)]
lt50 = [medianSCP(P[0], mpb.alpha)]
s = [1]
minT = [aphTmin[211]]
for t in range(t_days):
#first every time step we need to calculate the L and G values which are a function of tau and R
	tau = aphTmean[t]
	R = aphRange[t]
	if (C_t == 0):
		otG = mpb.mu_G
		otL = mpb.mu_L
	else:
		otG = mpb.optTemp(mpb.mu_G, mpb.kappa_G, C_t)
		otL = mpb.optTemp(mpb.mu_L, mpb.kappa_L, C_t)
	#gain and loss are defined by the same function, but have differing parameters
	L = mpb.changeInColdTol(R, tau, mpb.rho_L, mpb.sigma_L, otL)
	G = mpb.changeInColdTol(R, tau, mpb.rho_G, mpb.sigma_G, otG) 
	julien_date = t + 213
	delC = mpb.deltaC(G,L,C_t,julien_date)
	C_t = C_t + delC
	P_t = popProp(C_t, mpb.lambda_0, mpb.lambda_1, mpb.lambda_2)
	lt50.append(medianSCP(P_t, mpb.alpha))
	C_array.append(C_t)
	P.append(P_t)
	minT.append(aphTmin[t])
	s.append(survive(P, minT, mpb.alpha, mpb.beta))
	
C_array = np.array(C_array)
P = np.array(P)
lt50 = np.array(lt50)
s = np.array(s)
#cool it works,
#can we now see the states of the distribution of cold tolerance in real time?
#saved the plots, need to work on converting the figs to a movie
for t in range(365):
	plotPop(P[t], scpDist, C_array[t], testT, mpb.alpha, t)

'''
#things are looking good so far. So now let's determining the probability of survival. 
#looks like my proportion of the populations calculation was wrong. So I will use Regniere's version
def popProp2(C, lambda_0, lambda_1, lambda_2):
	p_1 = np.max(np.array([0,np.min((0.5-C)/(lambda_0-C))]))
	#not sure what this proportion is, but so far it is
	#how far is the current C from 0.5 (lambda_2) divided by how far C is from lambda_0
	#so lets imagine the inbetween point C= 0.377
	#that means the numerator is...0.123 and the denominator is -0.123 so we have -1
	#but this should be just inbetween so half should go to p1 and half to p2???
	#this doesn't make sense.
	#I think what I currently have coded makes much more sense, so I will keep that.
	p_3 = np.max(np.array([0,np.min((C-0.5)/(C-lambda_1))]))
	p_2 = 1 - (p_1 + p_3)
	popFac = np.array([p_1, p_2, p_3])
	return(popFac)
'''
#now solve for the median SCP value
def medianSCP(p, alpha):
	return(np.sum(p*alpha))

#that works, we should record the median SCP values #done
#now lets solve for the survival function
def survive(p, T, alpha, beta):
#easier with a for loop going backwards
	s = np.sum(p[-1]/(1+np.exp(-(T[-1] - alpha)/beta)))
	surv = np.min(1,s)
	for t in range(len(T)-2, 0, -1):
		s_t1 = np.sum(p[t-1]/(1+np.exp(-(T[t-1] - alpha)/beta)))
		surv = np.min(surv, s_t1)
	return(surv)
'''
C = np.array([0.1,mpb.lambda_0, 0.35,mpb.lambda_2,0.65,mpb.lambda_1])
for i in range(len(C)):
	popFac = popProp(C[i], mpb.lambda_0, mpb.lambda_1, mpb.lambda_2)
	plotPop(popFac, scpDist, C[i], testT)
'''