# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 22:19:23 2021

@author: billy huang
"""
#%%
import nltk
tagged_token = nltk.tag.str2tuple('fly/NN')
print(tagged_token)
#%%
sent = '''
    The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
    other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
    Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
    said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
    accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
    interest/NN of/IN both/ABX governments/NNS ''/'' ./.
    '''
print([nltk.tag.str2tuple(t) for t in sent.split()])
#%%
from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories='news',  tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
tag_fd.plot(cumulative=True)
#%%
# look for words that are highly ambiguous as to their part of speech tag
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
data = nltk.ConditionalFreqDist((word.lower(), tag)
    for (word, tag) in brown_news_tagged)
for word in sorted(data.conditions()):
    if len(data[word]) > 3:
        tags = [tag for (tag, _) in data[word].most_common()]
        print(word, ' '.join(tags))