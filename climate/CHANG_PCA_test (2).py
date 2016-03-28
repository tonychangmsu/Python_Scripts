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

def princomp(A):
 """ performs principal components analysis 
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
 # computing eigenvalues and eigenvectors of covariance matrix
 M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
 [latent,coeff] = la.eig(cov(M)) # attention:not always sorted
 score = dot(coeff.T,M) # projection of the data in the new space
 return coeff,score,latent
 
csv = "C:\\CHANG\\Climate_Models\\Station_Data\\Analysis\\StationsPCAformattedPython2_10152012.csv"  
csv2 = "C:\\CHANG\\Climate_Models\\Station_Data\\Analysis\\StationsPCAformattedPython_10152012.csv"
N = 321
K = 12
fraction = .90
seed = 1
#exec "\n".join( sys.argv[1:] )  # N= ...
np.random.seed(seed)
np.set_printoptions( 1, threshold=100, suppress=True )  # .1f
try:
    A = np.genfromtxt(csv, delimiter=",")
    N, K = A.shape
    r = mlab.csv2rec(csv2)
except IOError:
    A = np.random.normal( size=(N, K) )  # gen correlated ?
print "csv: %s  N: %d  K: %d  fraction: %.2g" % (csv, N, K, fraction)
#implementation 1
"""
mu = np.mean(A, axis=0)
C = np.corrcoef(A, rowvar =0)
eval, evec = la.eig(C)

covMatrix = np.mat(A).T * np.mat(A) # data' * data
covMatrix = np.divide(covMatrix, N-1) 

# covMatrix is a matrix type.  Now perform eigenvalue computation.
# v will contain eigenvalues and w contains vectors.
v,w = la.eig(covMatrix)

pyplot.plot(v) #skree plot
pyplot.show()
pyplot.plot(mlab.PCA(A))
"""
#implementation 2
"""
img = A
U,s, Vt = svd(A, full_matrices = False)
V = Vt.T

ind = np.argsort(s)[::-1]
U = U[:,ind]
s = s[ind]
V = V[:,ind]

# if we use all of the PCs we can reconstruct the noisy signal perfectly
S = np.diag(s)
Ahat = np.dot(U,np.dot(S,V.T))
print "Using all PCs, MSE = %.6G" %(np.mean((A-Ahat)**2))

# if we use only the first 12 PCs the reconstruction is less accurate
Ahat2 = np.dot(U[:,:12],np.dot(S[:12,:12],V[:,:12].T))
print "Using all 12 PCs, MSE = %.6G" %(np.mean((A-Ahat2)**2))
fig = pyplot.figure()
#pyplot.scatter(A)

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
Cooke = 42
Darwin = 78
Gardiner = 120
Jackson = 162
LakeYellowstone = 204
OldFaithful = 237
TowerFall = 278
YNPMammoth = 321

#PC,P,E = svd(A)
#pc1, pc2 = PC[:,0], PC[:,1]
#pyplot.scatter(pc1[0:1],pc2[0:1], marker = 'x', label = "winter")
#pyplot.scatter(pc1[10:12],pc2[10:12], marker = 'x')
#pyplot.scatter(pc1[1:4],pc2[1:4], marker = 'o', label = "spring")
#pyplot.scatter(pc1[4:7],pc2[4:7], marker = '^', label = "summer")
#pyplot.scatter(pc1[7:10],pc2[7:10], marker = '*', label = "fall")
#colors = cm.rainbow(np.linspace(0, 1, 12))
#for i in range(12):
#    pyplot.scatter(pc1[i],pc2[i], s = 60, marker = 's',c = colors[i], label = str(i+1))
   
'''
pyplot.scatter(pc1[0],pc2[0], c='blue', label = "jan")
pyplot.scatter(pc1[1],pc2[1], c='red', label = "feb")
pyplot.scatter(pc1[2],pc2[2], c='red', label = "mar")
pyplot.scatter(pc1[3],pc2[3], c='red', label = "apr")
pyplot.scatter(pc1[4],pc2[4], c='red', label = "may")
pyplot.scatter(pc1[5],pc2[5], c='red', label = "jun")
pyplot.scatter(pc1[6],pc2[6], c='red', label = "jul")
pyplot.scatter(pc1[7],pc2[7], c='red', label = "aug")
pyplot.scatter(pc1[8],pc2[8], c='red', label = "sep")
pyplot.scatter(pc1[9],pc2[9], c='red', label = "oct")
pyplot.scatter(pc1[10],pc2[10], c='red', label = "nov")
pyplot.scatter(pc1[11],pc2[11], c='red', label = "dec")

'''
[coeff, score, eigval] = princomp(A)
pc1 = score[0]
pc2 = score[1]
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
  
