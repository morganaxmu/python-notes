# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:43:50 2020

@author: billy huang
"""

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestCentroid

datapoints, labels = make_blobs(n_samples=300, centers=4, random_state=753, cluster_std=1.5)
for i in range(0,300):
    if labels[i]==1:
        labels[i]=3
plt.scatter(datapoints[:, 0], datapoints[:, 1], c=labels, s=50, cmap='rainbow');
plt.show()

# cross-validation Naive Bayes
model = GaussianNB()
model.fit(datapoints, labels)
scores = cross_val_score(model, datapoints, labels, cv=10)
print(scores.mean())

# Nearest Centroid model
classifier = NearestCentroid()
classifier.fit(datapoints, labels)
predictions = classifier.predict(datapoints)
print(accuracy_score(labels, predictions))

tree = DecisionTreeClassifier().fit(datapoints, labels)
scores = cross_val_score(tree, datapoints, labels, cv=10)
print(scores.mean())

# the decision tree is over-fit
print(scores)
predictions = tree.predict(datapoints)
print(accuracy_score(labels, predictions))


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
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
    
visualize_classifier(DecisionTreeClassifier(), datapoints, labels)