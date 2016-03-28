# Title: 		integration_stop.py
# Author: 		Tony Chang
# Date:			10.26.2015
# Abstract:		Attempt to find a solution to determining where the cumulative sum (numerical integration), of a array of 
				2D matricies sum up to one (find the index)
				
import numpy as np

#first suppose we have a 3D matrix of values under 1

G =  np.random.uniform(0,.05, (365,500,400))

#now develop a cumulative sum for each step

integral_G = np.cumsum(G, axis =0)

#now find out the index of the first axis where the value is equal to one.

index = np.argmax(integral_G>1, axis = 0)

#if any of these equals to 0 then we have a development that didn't complete, and we have a problem
#need more time to finish (i.e. more years to inspect). 

#done!



