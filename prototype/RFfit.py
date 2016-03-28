#CART analysis
import sklearn #import scikit-learn
import pylab
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
import csv

def write_csv(file_path, data):
    with open(file_path,"w") as f:
        for line in data: f.write(",".join(line) + "\n")

def read_csv(file, header = True):
	with open(file) as f:
		if header:
			f.readline()
		data= []
		for line in f:
			line = line.strip().split(",")
			#data.append([line])
			data.append([float(x) for x in line])
	return data

"""------------------------------------------------------------------------------
-------------------------MAIN----------------------------------------------------
------------------------------------------------------------------------------"""	

train = read_csv("D:\\chang\\python_scripts\\output\\WBPtrain03152013.csv", 'rb')
train = np.array(train)

y = train[:,9] #column of presence absence
x = train[:,13:29]

c = tree.DecisionTreeClassifier()
c = c.fit(x,y)

# with open("D:\\Chang\\Python_scripts\\output\\CTfit.dot",'w') as f:
#   f = tree.export_graphviz(clf3, out_file=f)
#import os
#os.unlink('CTfit.dot')