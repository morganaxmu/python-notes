# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 00:31:06 2021

@author: billy huang
"""

#%%
import nltk
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
print(unigram_tagger.tag(brown_sents[2007]),
      '\n',
      unigram_tagger.evaluate(brown_tagged_sents))
#%%
size = int(len(brown_tagged_sents) * 0.9)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
print(unigram_tagger.evaluate(test_sents))
#%%
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
print(t2.evaluate(test_sents))
#%%
#save tagger
from pickle import dump
output = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()
#load tagger
from pickle import load
input = open('t2.pkl', 'rb')
tagger = load(input)
input.close()