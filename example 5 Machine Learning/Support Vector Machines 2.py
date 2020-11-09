# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:35:33 2020

@author: billy huang
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
# load dataset
iris = sns.load_dataset('iris')

# Remember, we can only directly plot two attributes, e.g. sepal_length vs sepal_width
# Let's choose two to look at first (so we can easily change them later)
feature1 = 'sepal_length'
feature2 = 'petal_length'

species = iris['species'].unique()
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

for name in species:
    plt.scatter(iris[iris['species'] == name][feature1], iris[iris['species'] == name][feature2], c=colors[name])

plt.show()

datapoints = iris.drop('species', axis=1).values
labels = np.array(iris['species'])

datapoints = iris[[feature1, feature2]]
labels[labels == 'versicolor'] = 'virginica'

# grid search
# linear
model = SVC(kernel='linear')
parameters = {
              'C': [0.1, 1, 10, 100, 0.005],
             }
gridsearch = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
gridsearch.fit(datapoints, labels)
print(gridsearch.best_params_)
print(gridsearch.cv_results_['mean_test_score'])
print(gridsearch.cv_results_['params'])

# rbf
model = SVC(kernel='rbf')

parameters = {
              'C': [0.1, 1, 100, 1000],
              'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
             }
gridsearch = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
gridsearch.fit(datapoints, labels)
print(gridsearch.best_params_)

# poly
datapoints, labels = make_blobs(n_samples=300, centers=3, random_state=753, cluster_std=0.8)
for i in range(0,300):
    if labels[i]==0:
        labels[i]=1
model = SVC(gamma=1, C=0.1, kernel='poly')
model.fit(datapoints, labels)

rng = np.random.RandomState(0)
Xnew = [-8, -12] + [16, 12] * rng.rand(4000, 2)
ynew = model.predict(Xnew)
plt.figure(figsize=(20,15))
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='rainbow', alpha=0.1)
plt.scatter(datapoints[:, 0], datapoints[:, 1], c=labels, s=50, cmap='rainbow');
plot_svc_decision_function(model)