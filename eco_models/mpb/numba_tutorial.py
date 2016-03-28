#Title: numba_tutorial.py
#Author: Tony Chang
#Abstract: Learning to use numba for the first time to access GPU and speed up calculations
#Date: 08.17.2015

#import jit as a decorator to tell Numba to compile a particular function

from numba import jit 
from numba import vectorize
import numpy as np
import time
import math
#if we use the jit decorator, the argument types will be inferred by Numba when called
#"just in time' compiling...
@jit 
def sum2d(arr):
	M, N = np.shape(arr)
	result = 0
	for i in range(M):
		for j in range(N):
			result += arr[i,j]
	return(result)

def sum2d_non_numba(arr):
	M, N = np.shape(arr)
	result = 0.0
	for i in range(M):
		for j in range(N):
			result += arr[i,j]
	return(result)

a = np.arange(1000000).reshape(1000,1000)

timein = time.clock()
print(sum2d(a))
timeout = time.clock()
print('time is %s seconds' %(timeout-timein))

timein = time.clock()
print(sum2d_non_numba(a))
timeout = time.clock()
print('time is %s seconds' %(timeout-timein))


#we can use the threads of the cpu to speed things up...
#use the vectorize decorator to vectorize some calculation

#in the decorator we define two types of function outputs float32 and float64
#this is essentially a ufunc to target the CPU or GPU
#look up ufunc in the python tutorials
@vectorize(['float32(float32, float32)', 'float64(float64, float64)'], target = 'cpu') 
def cpu_sincos(x, y):
	return(math.sin(x) * math.cos(y))

x = np.random.random(1000000)
y = np.random.random(1000000)

timein = time.clock()
print(cpu_sincos(x,y))
timeout = time.clock()
print('time is %s seconds' %(timeout-timein))

#these vectorize functions are great but we are working with scalars in this example, what if I want to 
#work with array arguments?
'''
#use the Generalize Universal Function (guvectorize)
from numba import guvectorize
#note that we must define the descripte of the shape signature #note that float64 not safe and should not be used
@guvectorize(['void(float32[:,:], float32[:,:], float32[:,:])'],'(m,n),(n,p)->(m,p)', target='cpu')
def batch_matrix_mult(a,b,c):
	for i in range(c.shape[0]):
		for j in range(c.shape[1]):
			tmp = 0
			for n in range(a.shape[1]):
				tmp += a[i,n] * b[n,j]
			c[i,j] = tmp
	return(tmp)

n = 3
dim1 = 472
dim2 = 504
dim = 500
a = np.random.random(n * dim * dim).astype(np.float32).reshape(n, dim, dim)
b = np.random.random(n * dim * dim).astype(np.float32).reshape(n, dim, dim)
timein = time.clock()
cpu_ans = batch_matrix_mult(a,b)
print(cpu_ans)
timeout = time.clock()
print('time is %s seconds' %(timeout-timein))
'''

#let's try another jit
@jit('void(float32[:,:], float32, float32)')
def element_wise(u, x, y):
	#returns an nx,ny array of the two arrays
	nx, ny = u.shape
	for i in range(1,len(x)-1):
		for j in range(1,len(y)-1):
			u[i,j] = ((((u[i+1,j] + u[i-1,j]) * y) + ((u[i,j+1] + u[i, j-1]) * x))/ (2* (x+y))