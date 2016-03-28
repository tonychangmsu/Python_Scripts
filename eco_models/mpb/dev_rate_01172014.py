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
	y_raw = x
	y_raw = np.reshape(y_raw,(int(ndays),24)) # ndays X 24 hrs matrix of temperatures
	a = np.hstack((y_raw[1:,0],0)) #additional column to be added for endpoint
	y_raw = np.vstack((y_raw.T,a)).T #Add end point for integration
		#y = np.hstack(y(2:ndays,1) #Add end point for integration (original matlab code)
	y = np.delete(y_raw,-1,0) # elminate last day with no end point
		#y(ndays,:)=[] # elminate last day with no end point
	ndays=ndays-1 # set new size to number of days, for this case that should be 1459 days
	#dev_day = np.zeros((7,ndays))
	dev_day = []
	# produces an 8 no. life stages (rows) X ndays (columns) of daily developmental indices
	
	dev_day.append(np.trapz((blogan(y-5,pa[0,:])/24).T, axis=0)) #egg #finds the amount of development that has occurred at each day given temperature
	dev_day.append(np.trapz((blogan(y-5,pa[1,:])/24).T, axis=0)) #L1
	dev_day.append(np.trapz((blogan(y-10,pa[2,:])/24).T, axis=0)) #L2
	dev_day.append(np.trapz((typeiii(y,pa[3,:])/24).T, axis=0)) #L3
	dev_day.append(np.trapz((linear_rate(y,pa[4,:])/24).T, axis=0)) #L4
	dev_day.append(np.trapz((linear_rate(y,pa[5,:])/24).T, axis=0)) #pupae
	dev_day.append(np.trapz((stnrat(y,pa[6,:])/24).T, axis=0)) #Pre-ovipositional adult
	dev_day.append(np.trapz((gallery_ln(y,pa[7,:])/24).T, axis=0)) # Ovipositional adult
	return(np.array(dev_day), ndays)

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
	y=np.where(z3>0,z3,0)# filter for rates less than zero
	return(y)

def typeiii(t,p):
	xt=t-p[4]
	tau=(p[1]-xt)/p[2]
	x1=p[0]*(xt**2)/((xt**2)+(p[3]**2))
	x2=p[0]*(1-np.exp(-1*tau)) 
	x3=x1+x2-p[0]
	#z=np.reshape(np.where(x3>0, 1, 0), np.shape(x3))
	z=np.where(x3>0, 1, 0)
	q=np.where(t>p[4],1,0) #check for T less than base Temperature
	y=z*q*x3
	return(y)

def linear_rate(tmps,p):
	y=p[1]*(tmps-p[0])
	y[np.where(y<0)] = 0 #change all y values less than zero to 0
	return(y)

def stnrat(tmps,p):
# Stinners developmental rate curve
	t=tmps
	t[np.where(t>p[3])]= 2*p[3]-t[np.where(t>p[3])]
	y = p[0]/(1 + np.exp(p[1] + p[2] * t))
	y[np.where(y<0)]=0
	return(y)

def gallery_ln(tmps,p):
	'''
	function [y] = gall_ln(tmps,p)
	#	gallery length constructed as a function of temperature for mpb - see Logan et al. IUFRO 
	#	Hawaii Proceedings
	#	p(1) - P(3) are est parameters; p(4) is tau; p(5) is base Temperature
	'''
	t=tmps-p[4]
	t[np.where(t<0)]=0
	tau=t/p[3]
	y = ((p[0] * (np.exp(p[1]*t**p[2]) - np.exp(-tau))) * 2.54)/32 # the 2.54 converts from in to cm
	#divide by 32 to normalize to median
	return(y)

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
	ts=1 # starting value
	te=365 #ending value
	dt=1 # step size
	ntim=np.fix((te-ts)/dt)+1 # number of times
	count=0
	g = np.zeros(te)
	ret_line = [] 
	for i in range(ts,te+1):
		m_d_e=trap_devrats_new(d_d,ndays,i,1) #d_d must be pre-computed from dev_per_day
		print(m_d_e)
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
	% to get actual emergence date form any initial condition: (1) set
	% nyrs=1;  (2) comment out the modulo loop (3) set p_s=M_D_E(i,7)
	% (4) save x=tmp(:,3);
	%
	% compute the proportion of the life stage completed for each day in the temperature data set for each life stage in the model
	% simulation loop starts here
	'''
	
	#med_day_emerg= np.ones((nyrs,8))*np.nan
	med_day_emerg = np.zeros((nyrs,8))
	years_per_generation = np.zeros((nyrs,1))
	phase_space = np.zeros((nyrs,2))
	for i in range(nyrs):
		test=dev_day[0,(da_start-1):ndays].cumsum(axis=0)
		if (len(test) == 0): #check if test variable is empty?
			continue
		else:
			if(np.max(test)<1.0): #if there are no values in test that are greater than 1 exit the loop
				break
			med_day_emerg[i,0] = fnd1idx(test,1,1)+da_start+1 #initialize the first day stage median date of emergence
			#need to add 1 to da_start and fnd1idx to account for indexing at 0 
			for j in range(1,8): #else from 1 to 7 the different development stages
				test=dev_day[j,(med_day_emerg[i,j-1]-1):ndays].cumsum(axis=0) #find the index where the first cumsum is greater than 1
				if(np.max(test)<1.0):
					break
				med_day_emerg[i,j] = (med_day_emerg[i,j-1]+fnd1idx(test,1,1)+1) #initialize the first day stage median date of emergence
				
		#if(len(test) ==0 or np.max(test)<1.0):
		#	break
		#set beginning phase space 
		phase_space[i,0]=da_start
		years_per_generation[i,0]=(med_day_emerg[i,7]+1-da_start)/365 #med day emerg is off by one because indexing starts at 0, but actual years are by 1
		da_start=modulo(med_day_emerg[i,7]+1,365) # modulo reduction
		phase_space[i,1]=modulo(med_day_emerg[i,7]+1,365)
		if (da_start==0):
			da_start=1
		phase_space[i,1]=modulo(med_day_emerg[i,6]+1,365)
	return(med_day_emerg,phase_space,years_per_generation)
	
	
def modulo(x,y):
	'''
	# z = modulo = (x,y)
	# function to return the modulo of (x,y).  Where z is the remainder after dividing
	# x by y
	'''
	z = x-(np.floor(x/y)*y);
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
	if (len(x)==0):
		indx = 0
	elif (s>=0):
		indx=np.min(np.where(x>=v)[0]);
	else:
		indx=np.max(np.where(x<=v)[0]);
	return(indx)

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
	idx=fnd1idx(d,-1,-0.000001)
	plt.figure(0)
	plt.plot(return_line,return_line)
	plt.plot(return_line[0:idx],g[0:idx])
	#junk,idx2=np.shape(g)
	idx2 = len(np.where(g!=365)[0])-1
	plt.plot(np.arange(idx+1,idx2),g[idx+1:idx2])
	plt.plot([idx, idx+1],[g[idx], g[idx+1]],'r', ls = '--')
	plt.plot([1, idx2],[g[0], g[idx2]],'r', ls = '--')
	plt.show()
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
ts = -1.25 #starting value
te = 2.75 #ending value
dt = 0.25 #step size
ntim = np.fix((te-ts)/dt)+1 #number of times 
nyrs= 20

#load data files
workspace = 'D:\\chang\\phd_material\\wbp_project\\mpb_phenology\\data\\'
t = np.genfromtxt(workspace + 'temp.txt')  #t - hourly temperatures for 4 years?
pa = np.genfromtxt(workspace + 'p_new.dat') #p - array of developmental parameters for the trap_devrats_new model
t1 =[]
x0 = []
y0 = []
ntim =1
#for i in range(int(ntim)):
for i in range(ntim):
	to = ts+(i*dt)
	qt = t + to #modified temperature
	d_d,ndays = dev_per_day(qt,pa) #run the dev per day calculator...
	g, ret_line = G_curve(d_d,ndays)
	#m_d_e, p_s, y_p_g = trap_devrats_new(d_d, ndays, 240, nyrs)
	#x = np.ones(10)*to
	#t1.append(cycle_length(p_s[:,1]))
	#y = p_s[10:20,1]
	#p_s[:,1] = modulo(p_s[:,1],365)
	
	#print(str(i) + ' x:')
	#print(x)
	#print('\n')
	#print(str(i) + ' y:')
	#print(y)
	#print('\n')
	#x0.append(x[0])
	#y0.append(y[0])
	
	#plt.plot(x,y,'-')
	
#plt.show()
	#g,ret_line = G_curve(d_d,ndays)
	
#plot_g_function(g,ret_line)

#====================================#
