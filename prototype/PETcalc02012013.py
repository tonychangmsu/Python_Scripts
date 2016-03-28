# -*- coding: utf-8 -*-
"""
Created on Fri Feb 01 15:07:55 2013

@author: tony.chang
Climatic water deficit equations from 
Lutz, J.A., van Wagtendonk, J.W., and Franklin, J.F. Climatic water deficit, 
tree species, and climate change in Yosemite National Park. 2010. 
Journal of Biogeography

"""

#Initialization
"""Required variables to have for calculations
    Ta=         #mean monthly temperature
    Pm=         #mean monthly precipitation
    slope=      #slope of grid cell
    aspect=     #aspect of grid cell
    lat=        #latitude of grid cell
    sd=         #solar declination angle at noon on the 15th day of the month
    soilm=      #soil moisture values

Boundary conditions for site extent are
xmin = -110.948
xmax = -110.425
ymin = 43.537
ymax = 44.133

"""
import math

#test data
slope = linspace(0,(pi/2),7, endpoint=True)
aspect = linspace(0,2*pi, 360, endpoint=True) #the fold aspect only works from 0 - 180
#lat = linspace(43.537,44.133,90, endpoint=True)
lat = 44.133
#Defining monthly melt factor, a function of monthly temperature

def Meltfactor(Ta):
    if Ta <= 0:
        Fm = 0
    elif (Ta>0 or Ta<6):
        Fm = 0.167*Ta
    else:
        Fm =1
    return (Fm)
    
def Pack(Fm, Pm, Packprev):    
    return((1-Fm)**2 * Pm + (1-Fm) * Packprev)
    
def Waterinput(Fm, Pm, Packprev):
    rain_m = Fm*Pm
    snow_m = (1-Fm)*Pm
    pack_m = ((1-Fm)**2) * Pm + (1-Fm) * Packprev
    melt_m = Fm * (snow_m + Packprev)
    wm = rain_m + melt_m
    return(wm)

def HeatLI(lat,slope,af):
    HL = 0.339 + 0.808*(cos(lat)*cos(slope)) - 0.196*(sin(lat)*sin(slope)) - 0.482*(cos(af)*sin(slope))
    return(HL)
    
def PETcalc(ea,Ta,month,hl):
    av = 0.2618 #angular velocity of the Earth's rotation (rad/hr)    
    ea = 0.611*(exp(17.3*Ta)/(Ta+237.3)) #saturation vapour pressure
    dl = (2*cos-1(-tan(sd)*tan(lat)))/av #day length
    PET = 29.8*days*dl*hl*(ea*Ta/(Ta+273.2))
    return(PET)

def Soilcalc(soilmax,Wm,PET,soilm):  #calculate the soil moisture
    soilWHC = [soilmax, (Wm-PET)+soilm] #soilmax is the soil water holding capacity in the top 200cm of the soil profile    
    return(min(soilmax))

def Aspectf(aspect):
    #af = abs(radians(180)-abs((aspect)-radians(225)))
    af = abs(pi-abs((aspect)-(pi*5/4)))    
    return(af)

H=[]
for i in range(len(slope)):
    H.append(HeatLI(lat,slope[i],Aspectf(aspect)))

#visulizations
figure(1)
for i in range(len(H)):
    subplot(len(H), 1, i)
    ylabel('HI(slope=' + str(math.degrees(slope[i])) +")" )
    xlabel('aspect')
    plot(degrees(aspect),H[i])

    
    