import numpy
from numpy import *

##exercise in Iteratively reweighted least squares
##initial dataset
#pop = array([3929,5308,7239,9638,12866,17069,23191,31443,39818,50155,62947,75994,91972,105710,122775,131669,151325,179323,203211,226542,248710])
pop = array([5.59, 5.66,5.63,55.7,5.6])
year = arange(1780,1990,10)

MAD = median(abs(pop-median(pop))) #equivalent to standard deviation function 
MADyear = median(abs(year-median(year)))
x = year
y = pop
xtilde = median(year)
ytilde = median(pop)
xbar = mean(year)
ybar = mean(pop)
xstd = std(year)
ystd = std(pop)
r1 = zeros(len(pop)) #Least Median Square residuals
r2 = zeros(len(pop)) #MAD residuals
sxx = 0
sxy = 0
sxxmean = 0
sxymean = 0
cutoff = 2.5
#w1 = (1 - (x[i]/c)^2)^2 # bisquare weight function for |x| < c
w2 = 0 # bisquare weight function for all other x's
ko = 2.9366 #default k for asymptotically consistent scale estimate sigma with breakdown value of 25%
c = 1.483 # correction factor
S = c*MAD
T = ytilde
z = zeros(len(pop))
#STAGE 1
for i in range(0, len(pop)):
     sxx = sxx + ((x[i]-xtilde)*(x[i]-xtilde))
     sxy = sxy + ((y[i]-ytilde)*(x[i]-xtilde))
     sxxmean = sxxmean + ((x[i]-xbar)*(x[i]-xbar))
     sxymean = sxy + ((y[i]-ybar)*(x[i]-xbar))
     z[i] = (y[i] - T)/S # z-scores to identify outliers
betao = sxy/sxx # initial beta with breakdown 50%
betamean = sxymean/sxxmean
#STAGE 2
for i in range(0, len(pop)):
    r1[i] = y[i]-betao*x[i] #solve for residuals
    r2[i] = y[i]-MAD*x[i]
    # need to solve for the M-scale using bi-square function for error density

#STAGE 3
