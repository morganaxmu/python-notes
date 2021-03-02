# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:37:20 2021

@author: billy huang
"""
#%%
import nltk
puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
print([w for w in wordlist if len(w) >= 6
    and obligatory in w
    and nltk.FreqDist(w) <= puzzle_letters])
#%%
import nltk
entries = nltk.corpus.cmudict.entries()
p3 = [(pron[0]+'-'+pron[2], word)
    for (word, pron) in entries
    if pron[0] == 'P' and len(pron) == 3]
cfd = nltk.ConditionalFreqDist(p3)
for template in cfd.conditions():
    if len(cfd[template]) > 10:
        words = cfd[template].keys()
        wordlist = ' '.join(words)
        print(template, wordlist[:70] + "...")
#%%
from nltk.corpus import swadesh
fr2en = swadesh.entries(['fr', 'en'])
translate = dict(fr2en)
de2en = swadesh.entries(['de', 'en'])
es2en = swadesh.entries(['es', 'en'])
translate.update(dict(de2en))
translate.update(dict(es2en))
print(translate['chien'],translate['Hund'],translate['perro'])
