import numpy as np
from matplotlib import pyplot as plt

#==============DEVELOPMENT MODEL=============#
def dev_per_day(x,pa):
	'''
	Abstract
	program to compute the development per day for the trap_devrats_new model
	#	for temperatures in x - this is a helper function for trap_devrats_new
	#	note: one day will be lost due to requiring an end point for integration.  
	#	resulting in loosing one day of temperatures.   Therefore,
	#	you will need to add ond day of temperatures to the file, ie. for one year you need
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
	ndays=np.fix(len(x)/24) #find the number of days of date from the temperature data set
	y_raw = np.reshape(x,(int(ndays),24)) # ndays X 24 hrs matrix of temperatures
	a = np.hstack((y_raw[1:,0],0)) #additional column to be added for endpoint
	yy_raw = np.vstack((y_raw.T,a)).T #Add end point for integration
		#y = np.hstack(y(2:ndays,1) #Add end point for integration (original matlab code)
	yval = np.delete(yy_raw,-1,0) # eliminate last day with no end point
		#y(ndays,:)=[] # elminate last day with no end point
	#ndays=ndays-1 # set new size to number of days, for this case that should be 1459 days
	#dev_day = np.zeros((7,ndays))
	dev_day = []
	# produces an 8 no. life stages (rows) X ndays (columns) of daily developmental indices
	
	dev_day.append(np.trapz((blogan(yval-5,pa[0,:])/24).T, axis=0)) #egg #finds the amount of development that has occurred at each day given temperature
	dev_day.append(np.trapz((blogan(yval-5,pa[1,:])/24).T, axis=0)) #L1
	dev_day.append(np.trapz((blogan(yval-10,pa[2,:])/24).T, axis=0)) #L2
	dev_day.append(np.trapz((typeiii(yval,pa[3,:])/24).T, axis=0)) #L3
	dev_day.append(np.trapz((linear_rate(yval,pa[4,:])/24).T, axis=0)) #L4
	dev_day.append(np.trapz((linear_rate(yval,pa[5,:])/24).T, axis=0)) #pupae
	dev_day.append(np.trapz((strat(yval,pa[6,:])/24).T, axis=0)) #Pre-ovipositional adult
	dev_day.append(np.trapz((gallery_ln(yval,pa[7,:])/24).T, axis=0)) # Ovipositional adult insane bug in python?! need to assign this first
	
	dev_day = np.array(dev_day)
	return(dev_day, ndays) #output a percent development per day by life stage...

def blogan(tmps,p):
	'''
	function [y] = blogan(tmps,p)
	#	this function takes as input the vector 'temps' of temperatures and returns
	#	a vector y of the rate/temperature function 'alogan' value for each temperature in 
	#	temps, for parameters p1=psi, p2=increase rate, p3=max temperature, p4=del t 
	#	(high temp boundary)
	use:
	y=alogan(x,p)
	'''
	z1=1/(1+p[1]*np.exp(-p[2]*tmps))
	z2= np.exp(-(p[3]-tmps)/p[4])
	z3=p[0]*(z1-z2)
	#z4=z3[np.where(z3>0)] #original matlab code
	r=np.where(z3>0,z3,0)# filter for rates less than zero
	return(r)

def typeiii(t,p):
	xt=t-p[4]
	tau=(p[1]-xt)/p[2]
	x1=p[0]*(xt**2)/((xt**2)+(p[3]**2))
	x2=p[0]*(1-np.exp(-1*tau)) 
	x3=x1+x2-p[0]
	#z=np.reshape(np.where(x3>0, 1, 0), np.shape(x3))
	z=np.where(x3>0, 1, 0)
	q=np.where(t>p[4],1,0) #check for T less than base Temperature
	r=z*q*x3
	return(r)

def linear_rate(tmps,p):
	r=p[1]*(tmps-p[0])
	r[np.where(r<0)] = 0 #change all y values less than zero to 0
	return(r)

def strat(tmps,p):
# Stinners developmental rate curve
	t = np.where(tmps>p[3],(2*p[3]-tmps),tmps)
	r = p[0]/(1 + np.exp(p[1] + p[2] * t))
	v = np.where(r<0,0,r)
	return(v)

def gallery_ln(tmps,p):
	'''
	function [y] = gall_ln(tmps,p)
	#	gallery length constructed as a function of temperature for mpb - see Logan et al. IUFRO 
	#	Hawaii Proceedings
	#	p(1) - P(3) are est parameters; p(4) is tau; p(5) is base Temperature
	'''
	t=tmps-p[4]
	t = np.where(t<0, 0, t)
	tau=t/p[3]
	r = ((p[0] * (np.exp(p[1]*t**p[2]) - np.exp(-1*tau))) * 2.54)/32. # the 2.54 converts from in to cm
	#divide by 32 to normalize to median
	return(r)

#====================================G-FUNCTION======================================================
def G_curve(d_d,ndays):
	'''
	%
	%   Abstract;  Program to compute G-function for an annual temperature cycle
	%
	%   usage: Set up for p-new parameters
	%       [g,ret_line]=G_curve(p,d_d,ndays);
	%
	%   input
	%       d_d - the development per day for an observed annual temperature cycle
	%       ndays - number of days to run the model to get emergence - emergence date will be 
	%           reduced modulo 365
	%
	%   on output:
	%       g - G-function
	%       ret_line - 45 degree return line
	%
	%   see also trap_devrats_new, dev_per_day, plot_g_function, iterate_G
	% 
	'''
	ts=1 # starting value for days
	te=365 #ending value days
	dt=1 # step size
	ntim=np.fix((te-ts)/dt)+1 # number of times
	count=0
	g = np.zeros(te)
	ret_line = [] 
	for i in range(ts,te+1):
		m_d_e=trap_devrats_new(d_d,ndays,i,1) #d_d must be pre-computed from dev_per_day
		#print(m_d_e)
		g[count]=np.mod(m_d_e[0][0][7],365) # m_d_e(7) is median egg; m_d_e(6) is adult emergence
		count+=1
		ret_line.append(i)
	g[np.where(g==0)]=365 # avoid 0 dimension in circle map plotting
	return(g, np.array(ret_line))

def trap_devrats_new(dev_day,ndays,da_start,nyrs):
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
	%   Known Bugs - This function has not been rigoriously tested for very fast (<1 yr) or very slow
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
	med_day_emerg = np.zeros((nyrs+1,8+1))
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
	
def modulo(x,y):
	'''
	# z = modulo = (x,y)
	# function to return the modulo of (x,y).  Where z is the remainder after dividing
	# x by y
	'''
	z = x-(np.floor(x/y)*y)
	#could just use np.mod(x,y)
	return(z)

def fnd1idx(x,v,s):
	'''
	% index = find1index(x,v,s)
	%
	% this function returns an index to vector x as follows:
	% if s >= 0, then index is the smallest index s.t. x(index) >= v.
	% if s <= 0, then index is the smallest index s.t. x(index) <= v.
	%
	%  see also: FIND, SORT
	'''
	#if (len(x)==0):
	#	indx = 0
	if (s>=0):
		indx=np.min(np.where(x>=v)[0]);
	else:
		indx=np.max(np.where(x<=v)[0]);
	return(indx+1)

def plot_g_function(g,return_line):
	'''
	%
	% Abstract;  
	%   this program plots a g-function and the return line where
	%   g_function posses a fixed point.
	%
	% usage:
	%	plot_g_function(g,return_line)
	%
	%   input - these input arguments are output from a function like G-curve
	%       g = a g_function in with dim. (1Xk) wher g(n) is the result and development 
	%           milestone from the initial milestone in g(n-10.  max k = 365.
	%       return_line = coordinates for the 45 deg. line where resultant milestone 
	%           is identical to initial mile stone. Dim. (1Xk)
	%
	%   on output:
	%       g_function and return line will be ploted, no arguments will be returned
	%
	%   Known Bugs - 
	%
	%   see also:
	%       G_curve;  
	%       Logan & Powell.  2001. Am. Entomol 
	%
	'''	
	d=np.diff(g, n=1, axis=0) #approximate the derivative
	idx=fnd1idx(d,-1,-0.000001)-1
	plt.plot(return_line,return_line)
	plt.plot(return_line[0:idx],g[0:idx], color ='green')
	idx2 = len(np.where(g!=365)[0])-1
	plt.plot(np.arange(idx+1,idx2),g[idx+1:idx2], color = 'green')
	plt.plot([idx, idx+1],[g[idx], g[idx+1]],'r', ls = '--')
	plt.plot([1, idx2],[g[0], g[idx2]],'r', ls = '--')
	plt.grid()
	return(idx)
	
def cycle_length(tst):
	n=len(tst)-1
	z=np.where(tst==tst[n], tst, 0)
	cycle=np.diff(z, n=1, axis=0)
	if (len(cycle) == 0):
		y=np.inf
		return(y)
	ncycles=len(cycle)
	y=cycle[ncycles-1]
	return(y)

#==========================================================================================================#
#=============MAIN==================#
ts = -1.25 #starting value adjustment to the temperature dataset
te = 2.75 #ending value adjustment to the temperature dataset
dt = 0.25 #step size of temperature adjustments
ntim = np.fix((te-ts)/dt)+1 #number of simulations
nyrs= 1 # 4 year simulation periods

#load data files
workspace = 'E:\\MPB_model\\mpb_phenology\\data\\'
t = np.genfromtxt('%s%s' %(workspace, 'temp.txt'))  #t - hourly temperatures for 4 years?
pa = np.genfromtxt('%s%s' %(workspace, 'p_new.dat')) #p - array of developmental parameters for the trap_devrats_new model
t1 =[]
x0 = []
y0 = [] 
mt = [] #modified temperature array
ntim = ntim-1
ntim = 1
for i in range(int(ntim)):
	to = ts+(i*dt)
	qt = t + to #modified temperature, to emulate climate changes
	mt.append(qt)
	d_d,ndays = dev_per_day(qt,pa) #run the dev per day calculator...
	#need the output from dev_rate_new
	g, ret_line = G_curve(d_d,ndays)
	plt.subplot(4,4,i+1)
	plt.title('year '+ str(i+1) + '; mean temp = ' + str(np.mean(qt)))
	plot_g_function(g,ret_line)
	
plt.show()
#====================================#
