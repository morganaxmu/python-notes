# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:53:27 2021

@author: billy huang
"""
#%%
import nltk
def gender_features(word):
    return {'last_letter': word[-1]}
def gender_featuresl(word):
    return {'length': len(word)}
from nltk.corpus import names
import random
names = ([(name, 'male') for name in names.words('male.txt')] +
    [(name, 'female') for name in names.words('female.txt')])
random.shuffle(names)

featuresets = [(gender_featuresl(n), g) for (n,g) in names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.show_most_informative_features()
print(nltk.classify.accuracy(classifier, test_set))
#%%
print(featuresets)
#%%
from nltk.classify import apply_features
train_set = apply_features(gender_features, names[500:])
test_set = apply_features(gender_features, names[:500])