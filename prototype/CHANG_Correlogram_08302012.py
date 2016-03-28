"""Correlogram construction test
Programmer: Tony Chang
Course: BIOE 504 Quantitative Biology
Institution: Montana State University Bozeman
Instructor: Dr. Daniel Goodman

Description: This program will input a time series array and develop a correlogram
to evalute the autoregression type.

"""
import numpy
from numpy import *
import scipy
from scipy import *
import matplotlib

def Means(data):
    #input: time series array dataset
    #output: returns mean of dataset
    mu = 0
    for i in range(len(data)):
        mu = mu + data[i]
    return (mu/len(data))

def Variance(data):
    #input: time series array dataset
    #output: returns variance of dataset
    mu_hat = Means(data)
    n = len(data)
    sigmasq = 0
    for i in range(n):
        sigmasq = (data[i]**2) + sigmasq 
    sigmasq = (sigmasq/n) - mu_hat**2
    return(sigmasq)

def StandDev(data):
    #input: time series array dataset
    #output: returns standard deviation of dataset
    return (sqrt(Variance(data)))

def Covar(datax, datay):
    #input: time series array dataset and array of corresponding time values
    #output: returns covariance array
    cov = 0
    n = len(datax)
    #Numeric solution to cov = 1/n*(datax[i]-Means(datax))*(datay[i]-Means(datay))
    for i in range(len(datax)):
        cov = cov + datax[i]*datay[i]
    cov = (cov/n - ((Means(datax)*Means(datay))))
    return(cov)

def CorrMat(data):
    #input: time series array dataset and array of corresponding time values
    #output: returns correlation matrix
    n = len(data) #number of columns
    corrM = zeros((n,n))
    for i in range(n):
        for j in range (n):
            corrM[i,j] = Covar(data[i],data[j])/(StandDev(data[i])*StandDev(data[j]))
    return(corrM)

def CovMat(data):
    #input: time series array of n columns and m rows
    #output: returns the covariance matrix 
    n = len(data) #number of columns
    covM = zeros((n,n))
    for i in range(n):
        for j in range (n):
            covM[i,j] = Covar(data[i],data[j])
    return(covM)
    
def Corrgram(data):
    import matplotlib.pyplot as plt
    #input: time series array dataset
    #output: returns correlogram of lag n 
    n = len(data)
    cor_array = zeros(n)
    lag = zeros(n)
    
    for i in range(n):      #first loop through each column vector of matrix
        #for k in range(len(data[i))):   #second loop for each element of vector 
        cor_array[i] = Covar(data[:n-i],data[i:])
        lag[i] = i
        c = (data[:n-i],data[i:])
        """
        print 'Lag ', lag[i].astype(int), ' covariants' 
        print(CovMat(c))
        print 'correlation matrix:'
        print(CorrMat(c))
        print ''
        """
    y = cor_array[:n-1]
    x = lag[:n-1]
    plt.plot(x,y)
    return(x,y)


    
    
    
    
