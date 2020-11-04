# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:48:50 2020

@author: billy huang
"""

import sklearn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

iris_dataset = sns.load_dataset('iris')
# Remember, we can only directly plot two attributes in 2D, e.g. sepal_length vs sepal_width
# Let's choose two to look at first (putting their names into variables so we can easily change them later)
feature1 = 'sepal_length'
feature2 = 'petal_length'

species = iris_dataset['species'].unique()
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

for name in species:
    print(name)
    plt.scatter(iris_dataset[iris_dataset['species'] == name][feature1], iris_dataset[iris_dataset['species'] == name][feature2], c=colors[name])

plt.show()

iris = iris_dataset[iris_dataset['species'] != 'setosa']
for name in species:
    plt.scatter(iris[iris['species'] == name][feature1], iris[iris['species'] == name][feature2], c=colors[name])

plt.show()

classifier = NearestCentroid()

datapoints = iris.drop('species', axis=1).values

labels = np.array(iris['species'])
classifier.fit(datapoints, labels)
predictions = classifier.predict(datapoints)

print(accuracy_score(labels, predictions))

# The code in this cell sets our datapoints and labels to the data for the two classes 'versicolor' and 'virginica'
# This is so that we don't have too easy a problem!

# training and testing
iris = iris_dataset[iris_dataset['species'] != 'setosa']
datapoints = iris.drop('species', axis=1).values
labels = np.array(iris['species'])

for name in species:
    plt.scatter(iris[iris['species'] == name][feature1], iris[iris['species'] == name][feature2], c=colors[name])

plt.show()

data_train, data_test, labels_train, labels_test = train_test_split(datapoints, labels, test_size=0.5, random_state=1242)
classifier.fit(data_train, labels_train)

predictions_train = classifier.predict(data_train)
predictions_test = classifier.predict(data_test)

acc_train = accuracy_score(labels_train, predictions_train)
acc_test = accuracy_score(labels_test, predictions_test)

print("Accuracy on training data: ", acc_train)
print("Accuracy on test data: ", acc_test)

# Cross-validation
scores = cross_val_score(classifier, datapoints, labels, cv=10)
print(scores.mean())


## Multiple classes
iris = iris_dataset

# This is just a repeat of the code from earlier in the notebook, to plot an illustration of our data
# Remember: this plot only shows **two** features -- sklearn has access to all four
for name in species:
    plt.scatter(iris[iris['species'] == name][feature1], iris[iris['species'] == name][feature2], c=colors[name])

plt.show()
datapoints = iris.drop('species', axis=1).values
labels = np.array(iris['species'])
classifier.fit(datapoints, labels)
scores = cross_val_score(classifier, datapoints, labels, cv=5)
print(scores)
print(scores.mean())