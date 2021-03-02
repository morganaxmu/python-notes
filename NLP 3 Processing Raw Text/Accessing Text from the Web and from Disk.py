# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 22:22:46 2021

@author: billy huang
"""
#%%
from __future__ import division
import nltk, re, pprint
from urllib import urlopen
url = "http://www.gutenberg.org/files/2554/2554.txt"
raw = urlopen(url).read()
tokens = nltk.word_tokenize(raw)
#%%
url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = urlopen(url).read()
raw = nltk.clean_html(html)
tokens = nltk.word_tokenize(raw)
#%%
