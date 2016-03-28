# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:26:59 2013

@author: tony.chang
"""

def JuliandayConvert(M,D,Y):
    A = Y/100
    B = A/4
    C = 2-A+B
    E = 365.25*(Y+4716)
    F = 30.6001*(M+1)
    JD= C+D+E+F-1524.5
    return(JD)