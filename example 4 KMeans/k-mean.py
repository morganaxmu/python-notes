# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:20:27 2020

@author: billy huang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

dataset = pd.read_csv('./Mall_Customers.csv')
print(dataset.isnull().sum())

#Histograms
plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(dataset[x] , bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()

sns.pairplot(dataset[['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']])
plt.show


#Ploting the Relation between Age , Annual Income and Spending Score
#regression plot
plt.figure(1 , figsize = (15 , 7))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = dataset)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y ) #ternary conditional operator: a if condition else b
plt.show()

#scatter plot
plt.figure(1 , figsize = (15 , 6))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = dataset[dataset['Gender'] == gender] ,
                s = 200 , alpha = 0.5 , label = gender)
plt.xlabel('Age'), plt.ylabel('Annual Income (k$)') 
plt.title('Age vs Annual Income w.r.t Gender')
plt.legend()
plt.show()


#scatter plot
plt.figure(1 , figsize = (15 , 6))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,
                data = dataset[dataset['Gender'] == gender] ,s = 200 , alpha = 0.5 , label = gender)
plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)') 
plt.title('Annual Income vs Spending Score w.r.t Gender')
plt.legend()
plt.show()


#K-Means Clustering
#Segmentation using Age and Spending Score
'''Age and spending Score'''
X1 = dataset[['Spending Score (1-100)' , 'Annual Income (k$)']].iloc[: , :].values
inertia = []
for n in range(1 , 11): #n is the number of clusters
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)

#Selecting N Clusters based the Inertia (Squared Distance between Centroids and data points, should be less)
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()


#Redo the KMeans with optimal 4 clusters
algorithm = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_



#disply the clustering results with trainning data set
plt.scatter( x = 'Spending Score (1-100)' ,y = 'Annual Income (k$)' , data = dataset , c = labels1 , 
            s = 200 ) #drawn the real data with Scatter plot
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5) #draw the center
plt.ylabel('Annual Income (k$)') , plt.xlabel('Spending Score (1-100)')
plt.show() 



#display the clustering results with test data and their predictions
h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1

#generate the test set
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
#print(xx)
#print(yy)
#predict the result with testset
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

plt.figure(1 , figsize = (15 , 7) )
plt.clf() #Clear the current figure.
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower') 
#Display an image, i.e. data on a 2D regular raster.
#the backgound color part is drawn by the imshow function and Z

plt.scatter( x = 'Spending Score (1-100)' ,y = 'Annual Income (k$)' , data = dataset , c = labels1 , 
            s = 200 ) #drawn the real data with Scatter plot
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5) #draw the center
plt.ylabel('Annual Income (k$)') , plt.xlabel('Spending Score (1-100)')
plt.show() 
