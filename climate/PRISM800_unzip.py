# -*- coding: utf-8 -*-
"""
Created on Fri Mar 01 10:40:31 2013

@author: tony.chang
"""
import os
import tarfile

startyear = 1895
endyear = 2010
var = ["tmin", "tmax", "tdmean", "ppt", "tmean"]
workspace = "D:\\CHANG\\Climate_Models\\US_PRISM_800m\\uncompressed\\"
for i in range(len(var)):
    for j in range(startyear,endyear+1):
        filename = workspace + var[i] +"\\us_" + var[i] + "_" + str(j) + ".tar"
        t = tarfile.open(filename, 'r')
        t.extractall(workspace + "data\\" + var[i])
        t.close()
        
        