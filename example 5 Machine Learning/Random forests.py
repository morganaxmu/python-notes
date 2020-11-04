# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:35:31 2020

@author: billy huang
"""


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np

datapoints, labels = make_blobs(n_samples=300, centers=4, random_state=753, cluster_std=1.5)
for i in range(0,300):
    if labels[i]==1:
        labels[i]=3
plt.scatter(datapoints[:, 0], datapoints[:, 1], c=labels, s=50, cmap='rainbow')

decisiontree = DecisionTreeClassifier()
randomforest = BaggingClassifier(decisiontree, n_estimators=100, max_samples=0.8, random_state=1)

# How well would a single decision tree do?
scores_tree = cross_val_score(decisiontree, datapoints, labels, cv=10)
print("Single decision tree classifier scored: ", scores_tree.mean())

# How well does our random forest do?
scores_forest = cross_val_score(randomforest, datapoints, labels, cv=10)
print("Random forest of decision tree classifiers scored: ", scores_forest.mean())

def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1),
                           cmap=cmap,
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
    
visualize_classifier(decisiontree, datapoints, labels)
# visualize_classifier(randomforest, datapoints, labels)