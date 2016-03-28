# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:41:56 2013

@author: tony.chang
"""

import pycuda.driver as cuda
import pycuda.autinit, pycuda.compiler
import numpy as np

a = np.random.randn(4,4).astype(np.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu,a)

mod = pycuda.compiler.SourceModule("""     
    __global__ void twice(float *a)
    {
        int idx = threadldx.x + threadldx.y*4
        a[idx] *= 2
    }
    """) 
    #this takes the comment code into the cuda native C format
    
func = mod.get_function("twice")
func(a_gpu, block = (4,4,1)) #4 block 4 thread 

a_doubled = np.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print a_doubled
print a
