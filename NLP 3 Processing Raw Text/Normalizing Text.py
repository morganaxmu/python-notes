# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:05:54 2021

@author: billy huang
"""
#%%
import re
import nltk
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
    is no basis for a system of government. Supreme executive power derives from
    a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = nltk.word_tokenize(raw)
#%%
porter=nltk.PorterStemmer()
print([porter.stem(t) for t in tokens])
#%%
wnl = nltk.WordNetLemmatizer()
print([wnl.lemmatize(t) for t in tokens])