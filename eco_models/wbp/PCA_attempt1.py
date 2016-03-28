import numpy as np
from scipy.linalg import svd
from scipy.misc import lena
'''
#========================
#example
x = np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y = np.array([2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9])

data = np.array([x-np.mean(x),y-np.mean(y)])
cor = np.corrcoef(data)
cova = np.cov(data)
U, s, Vt = svd(data, full_matrices=False)
#=========================
'''
data = np.genfromtxt('E:\\wbp_model\\fielddata\\1950_1980_merged_data.csv', delimiter=',', names =True)

A = []
names = data.dtype.names
n = []
for i in names[3:]: #skip the first 3 columns representing lat, long, and response
	v = i
	n.append(v)
	A.append(data[i]-np.mean(data[i])) #use anomaly
n = np.array(n)
A = np.array(A)
U, s, Vt = svd(np.corrcoef(A), full_matrices=False)
V = Vt.T

#get the first PC index (first eigenvector)
ei = 0
ind = np.argsort(U[ei])[::-1]
PC = U[ei][ind]
#filter PC1 to correlations greater than 0.1
for i in range(len(U[ei])):
	print(n[ind][i], PC[i])

#pet8 PC1
#pet3 PC2
#pack5 PC3
#pet9 PC5
#pack3 PC7
#pet5 PC9
#soilm4 PC9
#pack1 PC11
#tmax7 PC12