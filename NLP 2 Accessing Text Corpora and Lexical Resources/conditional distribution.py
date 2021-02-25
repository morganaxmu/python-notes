# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 08:49:51 2021

@author: billy huang
"""

#%%
import nltk
from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))
genre_word = [(genre, word)
    for genre in ['news', 'romance']
    for word in brown.words(categories=genre)]
#%%
print(cfd.conditions())
#%%
from nltk.corpus import inaugural
cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target))
cfd.plot()
#%%
from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch',
    'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1'))
cfd.tabulate(conditions=['English', 'German_Deutsch'],
    samples=range(10), cumulative=True)
#%%
# Working with the news and romance genres from the Brown Corpus, find out 
# which days of the week are most newsworthy, and which are most romantic
days = ['Monday','Tuesday','Wednesday',"Thuesday",'Friday','Saterday','Sunday']
cfd = nltk.ConditionalFreqDist(
    (genre,words)
    for genre in ['news', 'romance']
    for words in brown.words(categories=genre)
    )
cfd.tabulate(samples=days)
cfd.plot(samples=days)