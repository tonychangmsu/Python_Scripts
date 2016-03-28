# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 06:12:17 2012

@author: Tony
"""
import numpy as np
from numpy import mean,cov,double, cumsum, dot,array, rank
import numpy.linalg as la
import matplotlib.pyplot as pyplot
from scipy.linalg import svd
import matplotlib.mlab as mlab
from matplotlib import cm as cm
from scipy import random
from mpl_toolkits.mplot3d import Axes3D

"""
def princomp(A):
    computing eigenvalues and eigenvectors of covariance matrix
    M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
    [latent,coeff] = la.eig(cov(M)) # attention:not always sorted
    score = dot(coeff.T,M) # projection of the data in the new space
    return coeff,score,latent
 

performs principal components analysis 
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to variables. 

    Returns :  
        coeff :
            is a p-by-p matrix, each column containing coefficients 
            for one principal component.
    score : 
    the principal component scores; that is, the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
    latent : 
    a vector containing the eigenvalues 
    of the covariance matrix of A.
"""
   
csv = "D:\\CHANG\\Climate_Models\\Station_Data\\Formatted Data\\CombinedPCA11012012.csv"  
yeartxt = "D:\\CHANG\\Climate_Models\\Station_Data\\Formatted Data\\PCAYears110112.csv"
N = 1022
K = 12
fraction = .90
seed = 1
A = np.genfromtxt(csv, delimiter=",")
yeararray = np.genfromtxt(yeartxt,delimiter=",")
N, K = A.shape

#implementation

# computing eigenvalues and eigenvectors of covariance matrix
M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
[latent,coeff] = la.eig(cov(M)) # attention:not always sorted
score = dot(coeff.T,M) # projection of the data in the new space

stationindex = [0,22,48,74,102,129,151,174,202,232,261,282,
                309,339,343,362,385,411,431,461,491,515,
                544,565,589,615,645,671,694,717,729,748,775,
                797,816,840,870,892,919,941,968,998,1021]
    
nameindex = ['Basecamp','Beartooth','BeaverCreek','Blackbear',
             'Blackwater','BoxCanyon','Canyon','CarrotBasin',
             'CookeCity','DarwinRanch','EveningStar','FisherCreek',
             'Gardiner','GrandTarghee','GraniteCreek','GrassyLake',
             'GrosVentreSummit','IslandPark','Jackson','LakeYellowstone',
             'LewisLake','LickCreek','MonumentPeak','MoranJunction',
             'NortheastEntrance','OldFaithful','ParkerPeak','PhillipsBench',
             'ShowerFalls','SnakeRiver','SnakeRiverStation','SylvanLake',
             'SylvanRoad','ThumbDivide','Togwotee','TowerFalls',
             'TwoOceanPlateau','WhiskeyCreek','WhiteMill','Wolverine',
             'YNPMammoth','YountsPeak']
             
elevationindex = [7333,9380,7851,8176,9865,6732,7887,9196,7460,8160,9068,9134,5280,9268,6814,7346,
                  8796,6319,6650,7835,7881,6896,8786,6798,7434,7320,9449,8084,8087,6896,6900,8491,7172,
                  8018,9610,6266,9281,6814,8700,7644,6300,8366]
pc1 = score[0]
pc2 = score[1]
pc3 = score[2]
fig = pyplot.figure()
"""
ax = fig.add_subplot(121,projection='3d')
for i in range(len(stationindex)-2):
    C1 = random.random()
    C2 = random.random()
    C3 = random.random()
    #pyplot.scatter(pc1[stationindex[i]:stationindex[i+1]],pc2[stationindex[i]:stationindex[i+1]], s = (elevationindex[i]/1000)**2, c = [[C1,C2,C3],[C1,C2,C3]])    
    p = ax.scatter(pc1[stationindex[i]:stationindex[i+1]],pc2[stationindex[i]:stationindex[i+1]],yeararray[stationindex[i]:stationindex[i+1]], 
                   s=(elevationindex[i]/3000)**4, c=str((elevationindex[i]/10000.)**2), marker = "o")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("year")
"""
ax2 = fig.add_subplot(111)
for i in range(len(stationindex)-2):
    p2 = ax2.scatter(pc1[stationindex[i]:stationindex[i+1]],pc2[stationindex[i]:stationindex[i+1]], 
                   s=(elevationindex[i]/3000)**3, c=str((elevationindex[i]/10000.)**2), marker = "o")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.grid()
ax2.show()
#ax.legend(loc=3, scatterpoints =1, prop={'size':12})
ax.grid()
ax.show()
fig.colorbar(p)
