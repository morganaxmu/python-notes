# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:27:49 2021

@author: billy huang
"""

#%%
import re
import nltk
word = 'supercalifragilisticexpialidocious'
regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
def compress(word):
    pieces = re.findall(regexp, word)
    return ''.join(pieces)
english_udhr = nltk.corpus.udhr.words('English-Latin1')
print(nltk.tokenwrap(compress(w) for w in english_udhr[:75]))

#%%
rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
cfd = nltk.ConditionalFreqDist(cvs)
cfd.tabulate()
cfd.plot()
#%%
cv_word_pairs = [(cv, w) for w in rotokas_words
                 for cv in re.findall(r'[ptksvr][aeiou]', w)]
cv_index = nltk.Index(cv_word_pairs)
print(cv_index['su'],cv_index['po'])

