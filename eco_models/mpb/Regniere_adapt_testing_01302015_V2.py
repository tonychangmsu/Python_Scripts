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
	
def plotPop(popFac, scpDist, C, T):
	for i in range(len(popFac)):
		plt.fill(T, popFac[i]*scpDist[i], label ='State %s'%(i+1), alpha = 0.5)
	plt.xlabel('Supercooling point')
	plt.ylabel('Simulated normalize frequency')
	plt.title('Supercooling state C: %s'%(C))
	plt.legend()
	plt.grid()
	plt.show()
	return()

####MAIN####
testT = np.linspace(-50,5,200)
scpState1 = mpb.SCP(mpb.alpha[0],mpb.beta[0], testT)
scpState2 = mpb.SCP(mpb.alpha[1],mpb.beta[1], testT)
scpState3 = mpb.SCP(mpb.alpha[2],mpb.beta[2], testT)
scpDist = np.array([scpState1, scpState2, scpState3])

#assign C here
'''
C = np.array([0.1,mpb.lambda_0, 0.35,mpb.lambda_2,0.65,mpb.lambda_1])
for i in range(len(C)):
	popFac = popProp(C[i], mpb.lambda_0, mpb.lambda_1, mpb.lambda_2)
	plotPop(popFac, scpDist, C[i], testT)
'''
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
aphTmax = phTmax[212:212+365]
aphTmean = phTmean[212:212+365]
aphRange = phTrange[212:212+365]

######################################################
