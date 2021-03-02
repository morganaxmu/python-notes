# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:01:11 2021

@author: billy huang
"""

#%%
import re
import nltk
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
edlist = [w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)]
print(edlist) 
#%%
word = ['liked','go','going','goes']
print([w for w in word if re.search('ed$|ing$|s$',w)])
#%%
print(re.findall(r'^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing'))
print(re.findall(r'^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing'))
print(re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing'))