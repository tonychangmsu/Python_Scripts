import numpy as np
import Regniere_adapt_01292015 as mpb
import matplotlib.pyplot as plt
'''
#test the distributions of the SCP functions
#temperatures in the paper run from -50 to 5 deg C
testT = np.linspace(-50,5,150)

scpState1 = mpb.SCP(mpb.alpha[0],mpb.beta[0], testT)
scpState2 = mpb.SCP(mpb.alpha[1],mpb.beta[1], testT)
scpState3 = mpb.SCP(mpb.alpha[2],mpb.beta[2], testT)

#plot all three SCPs?
plt.plot(testT, scpState1, label='State 1')
plt.plot(testT, scpState2, label='State 2')
plt.plot(testT, scpState3, label='State 3')
plt.legend()
plt.grid()
plt.show()

#the distribution function looks pretty good. but how does this apply to C? Do I randomly sample from these 3 distributions under a given
#phloem temperature?
#No. These represent the percent of the population in that state
#the state that you should sample from depends on C. So if C is 0.5, then 100% of the population is in State 2? What does that mean?
#
#let's think about it. These distribution functions derive from the field data. The field data shows the frequency of MPB and their 
#cold tolerance. So in other words these distributions show the proportion of cold tolerances for population at each state. For example, 
#if the entire population was at state 2 (ie C = 0.5) then 17% of the population has a supercooling point of -21.2 deg C. Okay that makes sense
#now what? So next we need to determine what percent of the population is in each state. i.e. if 1/3 of the population is in each state,
#then we know the mean supercooling temperature for all individuals

#lets start by assuming the population has 100 individuals
pop = 100
#assume the population is divided into each state
pop1 = np.round(pop*(2/7))
pop2 = np.round(pop*(4/7))
pop3 = pop-pop1-pop2

#plot all three SCPs?
frqState1 = pop1*scpState1
frqState2 = pop2*scpState2
frqState3 = pop3*scpState3
plt.fill(testT, frqState1, label='State 1', alpha = 0.7)
plt.fill(testT, frqState2, label='State 2', alpha = 0.7)
plt.fill(testT, frqState3, label='State 3', alpha = 0.7)
plt.xlabel('Supercooling point')
plt.ylabel('Simulated frequency')
plt.legend()
plt.grid()
plt.show()

##############################
######State1-2 example########
##############################
#great, so now we know the threshold when the population is completely state 1, state 2, or state 3 
#all State 1: C <= lambda_0 = 0.254
#all State 2: C >= 0.5 
#all State 3: C >= lambda_1 = 0.764  
#so given this understanding does that mean that the C provides us with the percentage of the total population in each state?
#furthermore, this assumes that the population can only exist in two states at any given time for each point location. Yes, this must be 
#the case. Under the assumptions of Regneire. Beyond 2 point locations, the population could be otherwise. 
#so C represents the total amount of the population in each state
#lets test
pop = 1
dif_lambda = mpb.lambda_2-mpb.lambda_0 #total range of C from 100%
C = .39
state2Prop = (C-mpb.lambda_0)/dif_lambda
state1Prop = 1-state2Prop
state3Prop = 1-(state2Prop+state1Prop)
#so now we have the factor variable
pop1 = pop*(state1Prop)
pop2 = pop*(state2Prop)
pop3 = pop*(state3Prop)
#plot all three SCPs?
frqState1 = pop1*scpState1
frqState2 = pop2*scpState2
frqState3 = pop3*scpState3
plt.fill(testT, frqState1, label='State 1', alpha = 0.7)
plt.fill(testT, frqState2, label='State 2', alpha = 0.7)
plt.fill(testT, frqState3, label='State 3', alpha = 0.7)
plt.xlabel('Supercooling point')
plt.ylabel('Simulated normalized frequency')
plt.legend()
plt.grid()
plt.show()

##############################
######State2-3 example########
##############################
#okay so we should do this for state 3 and figure out the pattern such that we can generate an if then case for state3 proportion
pop = 1
C = 0.7
dif_lambda = mpb.lambda_1-mpb.lambda2
state3Prop = (C-mpb.lambda_s2)/dif_lambda
state2Prop = 1-state3Prop
state1Prop = 1-(state2Prop+state3Prop)
#so now we have the factor variable
pop1 = pop*(state1Prop)
pop2 = pop*(state2Prop)
pop3 = pop*(state3Prop)
#plot all three SCPs?
frqState1 = pop1*scpState1
frqState2 = pop2*scpState2
frqState3 = pop3*scpState3
plt.fill(testT, frqState1, label='State 1', alpha = 0.7)
plt.fill(testT, frqState2, label='State 2', alpha = 0.7)
plt.fill(testT, frqState3, label='State 3', alpha = 0.7)
plt.xlabel('Supercooling point')
plt.ylabel('Simulated normalize frequency')
plt.legend()
plt.grid()
plt.show()
state1Prop = 1-state2Prop
#####################################
'''
#so let's build the function that sets the populations
#first off we define the functionm

def popProp(C, lambda_0, lambda_1, lambda_2):
	dif_lambda1 = lambda2 - lambda_0  
	dif_lambda2 = lambda_1 - lambda_2
	popFac = np.zeros(3)
	if (C<=lambda_0): #first case, 100% in stage 1
		popFac[0] = 1
	elif (C<lambda_2): #second case, stage 1-2 transition
		popFac[1] = (C-lambda_0)/dif_lambda
		popFac[0] = 1-popFac[1]
	elif (C<lambda_1):
		popFac[2] = (C-lambda_2)/dif_lambda
		popFac[1] = 1-popFac[2]
	else: #C >=lambda_1
		popFac[2] = 1
	return(popFac)
	
def plotPop(popFac, scpDist, T):
	for i in range(len(popFac)):
		plt.fill(T, popFac[i]*scpDist[i], label ='State %s'%(i), alpha = 0.5)
		plt.xlabel('Supercooling point')
		plt.ylabel('Simulated normalize frequency')
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
C = 0.25
popFac = popProp(C, mpb.lambda_0, mpb.lambda_1, mpb.lambda_2)
plotPop(popFac, scpDist, testT)


