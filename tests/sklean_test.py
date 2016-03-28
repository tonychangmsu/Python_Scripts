# -*- coding: utf-8 -*-
"""
Created on Thu Mar 07 09:15:12 2013

@author: tony.chang
"""
#dataset can be found at https://github.com/jakevdp/scikit-learn/tree/master/sklearn/datasets
#or go to jakevdp.github.com/tutorial/astronomy/general_concepts.html for a full tutorial
from matplotlib import pyplot as pp

class ABLine2D(pp.Line2D):
    """
    Draw a line based on its slope and y-intercept. Keyword arguments are
    passed to the <matplotlib.lines.Line2D> constructor.
    """

    def __init__(self,slope,intercept,**kwargs):

        # get current axes if user has not specified them
        ax = kwargs.pop('axes',pp.gca())

        # if unspecified, get the line color from the axes
        if not (kwargs.has_key('color') or kwargs.has_key('c')):
            kwargs.update({'color':ax._get_lines.color_cycle.next()})

        # init the line, add it to the axes
        super(ABLine2D,self).__init__([None],[None],**kwargs)
        self._slope = slope
        self._intercept = intercept
        ax.add_line(self)

        # cache the renderer, draw the line for the first time
        ax.figure.canvas.draw()
        self._update_lim(None)

        # connect to axis callbacks
        self.axes.callbacks.connect('xlim_changed',self._update_lim)
        self.axes.callbacks.connect('ylim_changed',self._update_lim)

    def _update_lim(self,event):
        """ called whenever axis x/y limits change """
        x = np.array(self.axes.get_xbound())
        y = (self._slope*x)+self._intercept
        self.set_data(x,y)
        self.axes.draw_artist(self)
        
import sklearn #import scikit-learn
import pylab
import numpy as np
import matplotlib as plt
from sklearn.datasets import load_iris
data = load_iris() #load the iris dataset
"This is the general formatting routine"
x = data.data #this loads the feature data from the iris dataset
y = data.target #this is the class for each iris, we can use this format or use a 3 column vector with a binomial classification
name = data.target_names #classification labels
scatter(x[:,0], x[:,1], c=y) #general plot to see data for first 2 dimensions

#let's use support vector machine!
from sklearn.svm import LinearSVC
clf = LinearSVC() #instantiate the class clf = classifier
clf.fit(x,y) #fit the model
clf.coef_ #display the coefficients, (parameters)
clf.intercept_ #display the intercepts

x_new = np.array([[5.0,3.6,1.3,0.25]]) #this is our new data point we want to classify (these are the feature values)
clf.predict(x_new)
#predicts the new point (it is 0 which refers to y=0)
print name

#let's try a logistic regression fit!
from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression().fit(x,y)
clf2.predict_proba(x_new)
#will display the probabilities of each class
