# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:20:16 2013

@author: tony.chang
"""

import pyopencl as cl
import numpy as np
import numpy.linalg as la

'''
a = numpy.random.rand(5000).astype(numpy.float32)
b = numpy.random.rand(5000).astype(numpy.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

prg = cl.Program(ctx, """
    __kernel void sum(__global const float *a,
    __global const float *b, __global float *c)
    {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
    """).build()

prg.sum(queue, a.shape, None, a_buf, b_buf, dest_buf)

a_plus_b = numpy.empty_like(a)
cl.enqueue_copy(queue, a_plus_b, dest_buf)

print la.norm(a_plus_b)
'''
n = 100
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

a_gpu = cl.array.to_device(ctx, queue, np.random.randn(n).astype(np.float32))
b_gpu = cl.array.to_device(ctx, queue, np.random.randn(n).astype(np.float32))

from pyopencl.elementwise import ElementwiseKernel
lin_comb = ElementwiseKernel(ctx, "float a, float *x, float b, float *y, float*z", 
                             "z[i] = a*x[i] + b* y[i]")
                             
c_gpu = cl.array.empty_like(a_gpu)
lin_comb(5, a_gpu, 6, b_gpu, c_gpu)

assert la.norm((c_gpu - (5*a_gpu+ 6*b_gpu)).get()) < 1e-5
                            