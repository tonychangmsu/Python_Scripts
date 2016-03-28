# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 10:53:10 2013

@author: tony.chang
"""

# -*- coding: utf-8 -*--------------------------------------------------------
"""
Created on Wed Jan 30 10:12:42 2013

@author: tony.chang
Extract PNA data and plot seasonal and annual indices
"""
import numpy as np
import matplotlib.pyplot as plt
#initialize-------------------------------------------------------------------
"""import teleconnection data patterns"""
Index = np.genfromtxt('D:\CHANG\Climate_models\globalpatterns\PNA\PNAMonthly_1950_2012.txt', names=True)

def runningMean(x,N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
        y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N
    
#body------------------------------------------------------------------------
"""gather columns for season trends"""
data = []
year = []
for i in range(len(Index)-1):
    for j in range(1,12):    
        data.append(Index[i][j])
        year.append(Index[i][0]+(j/12.))
z = np.zeros(len(data))

winter =[]
summer = []
spring = []
fall = []
syear = []

for i in range(len(Index)-1):
    winter.append((Index[i][1]+Index[i][2]+Index[i][3])/3)
    spring.append((Index[i][4]+Index[i][5]+Index[i][6])/3)
    summer.append((Index[i][7]+Index[i][8]+Index[i][9])/3)
    fall.append((Index[i][10]+Index[i][11]+Index[i][12])/3)    
    syear.append(Index[i][0])
'''
for i in range(len(Index)-1):
    spring.append((Index[i][3]+Index[i][4]+Index[i][5])/3)
    summer.append((Index[i][6]+Index[i][7]+Index[i][8])/3)
    fall.append((Index[i][9]+Index[i][10]+Index[i][11])/3)    
    if i == 0:    
        winter.append((Index[i][1]+Index[i][2])/2)
    else:
        winter.append((Index[i-1][12]+Index[i][1]+Index[i][2])/3)
    syear.append(Index[i][0])
'''
sdata=[winter,summer,spring,fall]
zz = np.zeros(len(syear))

datarm = runningMean(data,10)
#plots------------------------------------------------------------------------
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(year,data, color = 'b', linewidth = .5)
ax.plot(year,datarm, color = 'black', linestyle ='--')
ax.grid(True)
ax.fill_between(year,z,datarm, where=datarm>z, facecolor = 'red',interpolate= True, alpha = .6)
ax.fill_between(year,z,datarm, where=datarm<z, facecolor = 'blue', interpolate= True, alpha = .6)
plt.xlabel('Year')
plt.ylabel('Standard Departure')
plt.title('PNA indices')

plt.figure(2)
labels = ['winter', 'spring', 'summer', 'fall']
for i in range(len(sdata)):
    subplot(4,1,i+1)
    plot(syear,sdata[i], color = 'black', linewidth = .1, linestyle='--')
    grid(True)
    fill_between(syear, zz, sdata[i], where=sdata[i]>zz, facecolor = 'red', interpolate = True)
    fill_between(syear, zz, sdata[i], where=sdata[i]<zz, facecolor = 'blue', interpolate = True)
    ylabel(labels[i] + " Index")