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
   
csv = "C:\\CHANG\\Climate_Models\\Station_Data\\Formatted Data\\CombinedPCA11012012.csv"  
N = 1022
K = 12
fraction = .90
seed = 1

A = np.genfromtxt(csv, delimiter=",")
N, K = A.shape

#implementation

# computing eigenvalues and eigenvectors of covariance matrix
M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
[latent,coeff] = la.eig(cov(M)) # attention:not always sorted
score = dot(coeff.T,M) # projection of the data in the new space

"""
#implementation 3
coeff, score, latent = princomp(A.T)

pyplot.figure()
pyplot.subplot(121)
# every eigenvector describe the direction
# of a principal component.
m = mean(A,axis=1)
pyplot.plot([0, -coeff[0,0]*2]+m[0], [0, -coeff[0,1]*2]+m[1],'--k')
pyplot.plot([0, coeff[1,0]*2]+m[0], [0, coeff[1,1]*2]+m[1],'--k')
pyplot.plot(A[0,:],A[1,:],'ob') # the data
pyplot.axis('equal')
pyplot.subplot(122)
# new data
pyplot.plot(score[0,:],score[1,:],'*g')
pyplot.axis('equal')
pyplot.show()
"""

# implementation 4
stationindex = [22,48,74,102,129,151,174,202,232,261,282,
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

"""
pyplot.scatter(pc1[:Cooke],pc2[:Cooke],s=60, c = 'red', marker= 'o', label = "Cooke, 7460ft")
pyplot.scatter(pc1[Cooke:Darwin],pc2[Cooke:Darwin],s=60, c = 'blue', marker='x', label = "Darwin, 8160ft")
pyplot.scatter(pc1[Darwin:Gardiner],pc2[Darwin:Gardiner],s=60, c = 'green', marker='+', label = "Gardiner, 5305ft")
pyplot.scatter(pc1[Gardiner:Jackson],pc2[Gardiner:Jackson],s=60, c = 'yellow', marker='*', label = "Jackson, 6450ft")
pyplot.scatter(pc1[Jackson:LakeYellowstone],pc2[Jackson:LakeYellowstone], s=60,c = 'purple', marker='d', label = "LakeYellowstone, 7835ft")
pyplot.scatter(pc1[LakeYellowstone:OldFaithful],pc2[LakeYellowstone:OldFaithful], c = 'black', marker='^', label = "OldFaithful, 7320ft")
pyplot.scatter(pc1[OldFaithful:TowerFall],pc2[OldFaithful:TowerFall], s=60,c = 'grey', marker='v', label = "TowerFalls, 6266ft")
pyplot.scatter(pc1[TowerFall:YNPMammoth],pc2[TowerFall:YNPMammoth],s=60, c = 'pink', marker='s', label = "YNPMammoth, 6300ft")

pyplot.ylabel("PC2")
pyplot.xlabel("PC1")
pyplot.legend(loc=3, scatterpoints =1, prop={'size':12})
pyplot.grid()
pyplot.show()

"""