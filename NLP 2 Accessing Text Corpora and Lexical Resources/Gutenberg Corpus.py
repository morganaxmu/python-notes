# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:20:38 2021

@author: billy huang
"""

import nltk
nltk.corpus.gutenberg.fileids()
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
print(len(emma),end='\n')
