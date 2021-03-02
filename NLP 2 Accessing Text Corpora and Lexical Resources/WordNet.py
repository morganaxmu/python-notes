# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 19:25:26 2021

@author: billy huang
"""

#%%
from nltk.corpus import wordnet as wn
print(wn.synsets('motorcar'))
print(wn.synset('car.n.01').lemma_names(),'\n',
      wn.synset('car.n.01').definition(),'\n',
      wn.synset('car.n.01').examples(),'\n',
      wn.synset('car.n.01').lemmas())
#%%
print(wn.lemma('car.n.01.automobile'),'\n',
      wn.lemma('car.n.01.automobile').synset(),'\n',
      wn.lemma('car.n.01.automobile').name(),'\n',
      wn.lemmas('car'))
#%%
print(dir(wn.synset('harmony.n.02')))