# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:47:45 2021

@author: billy huang
"""
#%%
import nltk
text = nltk.word_tokenize("And now for something completely different")
print(nltk.pos_tag(text))
#%%
text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
print(text.similar('woman'))