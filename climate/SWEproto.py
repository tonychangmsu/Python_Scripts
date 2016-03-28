# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 12:31:59 2013

@author: tony.chang
"""

#PYTHON CODE:
# -*- coding: utf-8 -*-
import numpy as np
def Dateformat(date):
    m = (date-(date%10000))/10000
    y = date%100
    d = ((date%10000)-y)/100
    if (y<=13):
        y = y+2000
    else:
        y = y+1900
    return(m,d,y)

SWEarray = [] #12x1 array
zerosnowarray = []
zerosnowday =[]
meltarray = []
annualarray = []
SWE = [0,1,2,1,2,2,3,4,0,1,1,2,1]
date = [101212,101312,110112,110212,12913, 13013, 13113, 20113, 20213, 20313, 20413, 20513,102513]

zerosnow = 0
positive = 0
melt = 0


for i in range(1,len(date)):
    m,d,y = Dateformat(date[i])
    mp,dp,yp = Dateformat(date[i-1])
    if (y==yp):
        if (m==mp):
            dSWE = SWE[i]- SWE[i-1]
            if (SWE[i-1] == 0):
                zerosnow = zerosnow + 1         #counts days without snow
                zerosnowarray.append(date[i-1])
            if (dSWE >=0):
                positive = positive + dSWE        #adds new snow counter
            else:
                melt = melt + dSWE        #adds melt counter
        else:
            dSWE = SWE[i]- SWE[i-1]
            if (SWE[i-1] == 0):
                zerosnow = zerosnow + 1         #counts days without snow
                zerosnowarray.append(date[i-1])
            if (dSWE >=0):
                positive = positive + dSWE        #adds new snow counter
            else:
                melt = melt + dSWE        #adds melt counter
            SWEarray.append(positive)
            meltarray.append(melt)
            zerosnowday.append(zerosnow)
            positive = 0
            melt = 0
            zerosnow = 0
        if (i== len(date)-1):
            SWEarray.append(positive)
            meltarray.append(melt)
            zerosnowday.append(zerosnow)
    else:
        dSWE = SWE[i]- SWE[i-1]
        if (SWE[i-1] == 0):
            zerosnow = zerosnow + 1         #counts days without snow
            zerosnowarray.append(date[i-1])
        if (dSWE >=0):
            positive = positive + dSWE        #adds new snow counter
        else:
            melt = melt + dSWE        #adds melt counter
        SWEarray.append(positive)
        meltarray.append(melt)
        zerosnowday.append(zerosnow)
        annualarray.append([yp,np.array(SWEarray),np.array(meltarray),np.array(zerosnowday),np.array(zerosnowarray)])
        SWEarray = [] #12x1 array
        zerosnowarray = []
        zerosnowday =[]
        meltarray = []
        positive = 0
        melt = 0
        zerosnow = 0
    if (i== len(date)-1):
        annualarray.append([yp,np.array(SWEarray),np.array(meltarray),np.array(zerosnowday),np.array(zerosnowarray)])