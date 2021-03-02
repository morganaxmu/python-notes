# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 22:42:08 2021

@author: billy huang
"""
import nltk
fdist = nltk.FreqDist(['dog', 'cat', 'dog', 'cat', 'dog', 'snake', 'dog', 'cat'])
for word in fdist:
    print('%s->%d;' % (word, fdist[word]))
#%%
def tabulate(cfdist, words, categories):
    print('%-16s' % 'Category',)
    for word in words: # column headings
        print('%6s' % word,)
    print
    for category in categories:
        print('%-16s' % category,) # row heading
        for word in words: # for each word
            print('%6d' % cfdist[category][word],) # print table cell
        print # end the row
from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
print(tabulate(cfd, modals, genres))
# different output...
#%%
output_file = open('output.txt', 'w')
words = set(nltk.corpus.genesis.words('english-kjv.txt'))
for word in sorted(words):
    output_file.write(word + "\n")