# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:12:42 2013

@author: tony.chang
"""
import numpy as np
import matplotlib.pyplot as plt
#initialize
"""import teleconnection data patterns"""
MEI = np.genfromtxt('D:\CHANG\Climate_models\globalpatterns\MEI\MEI_indexmonthly.txt', names=True)

#body
"""gather columns for season trends"""
data = []
year = []
for i in range(len(MEI)-1):
    for j in range(1,12):    
        data.append(MEI[i][j])
        year.append(MEI[i][0]+(j/12.))
z = np.zeros(len(data))

winter =[]
summer = []
spring = []
fall = []
syear = []

for i in range(len(MEI)-1):
    spring.append((MEI[i][3]+MEI[i][4]+MEI[i][5])/3)
    summer.append((MEI[i][6]+MEI[i][7]+MEI[i][8])/3)
    fall.append((MEI[i][9]+MEI[i][10]+MEI[i][11])/3)    
    if i == 0:    
        winter.append((MEI[i][1]+MEI[i][2])/2)
    else:
        winter.append((MEI[i-1][12]+MEI[i][1]+MEI[i][2])/3)
    syear.append(MEI[i][0])

sdata=[winter,summer,spring,fall]
zz = np.zeros(len(syear))


#plots
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(year,data, color = 'b', linewidth = .5)
ax.grid(True)
ax.fill_between(year,z,data, where=data>z, facecolor = 'red',interpolate= True, alpha = .6)
ax.fill_between(year,z,data, where=data<z, facecolor = 'blue', interpolate= True, alpha = .6)
plt.xlabel('Year')
plt.ylabel('Standard Departure')
plt.title('Multivariate ENSO indices')
plt.show(fig)

fig2 = plt.figure(2)
labels = ['winter', 'spring', 'summer', 'fall']
for i in range(len(sdata)):
    plt.subplot(4,1,i+1)
    plt.plot(syear,sdata[i], color = 'black', linewidth = .1, linestyle='--')
    plt.grid(True)
    plt.fill_between(syear, zz, sdata[i], where=sdata[i]>zz, facecolor = 'red', interpolate = True)
    plt.fill_between(syear, zz, sdata[i], where=sdata[i]<zz, facecolor = 'blue', interpolate = True)
    plt.ylabel(labels[i] + " MEI")
plt.show(fig2)