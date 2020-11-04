# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:00:38 2020

@author: billy huang
"""
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
from os import listdir
from os.path import isfile, join

def fetch_text_files(foldername):
    list_of_text = []
    for f in listdir(foldername):
        filepath = join(foldername, f)
        if isfile(filepath):
            this_text = ""
            with open(filepath, 'r') as file:
                this_text = file.read().replace('\n', '')
            list_of_text.append(this_text)
    return list_of_text

negative = fetch_text_files(join("review_polarity", "txt_sentoken", "neg"))
positive = fetch_text_files(join("review_polarity", "txt_sentoken", "pos"))

# Combine our negative and positive texts into one Python dictionary variable, adding labels "negative" and "positive":
data = {'text': negative + positive, 'class': ["negative"] * len(negative) + ["positive"] * len(positive)}

# Then use this to create a Pandas DataFrame so we can read it more easily
dataset = pd.DataFrame(data)
vec = TfidfVectorizer()
tf_values = vec.fit_transform(dataset['text'])
vector_data = pd.DataFrame(tf_values.toarray(), columns=vec.get_feature_names())
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(data['text'], data['class'])

predictions = model.predict(vector_data)
print(predictions[0])
my_review = pd.DataFrame(['I like it, it is awesome'], columns=['text'])
print(model.predict(my_review))

data_train, data_test, labels_train, labels_test = train_test_split(data['text'], data['class'], test_size=0.25, random_state=42)
model.fit(data_train, labels_train)
predictions = model.predict(data_test)
print(accuracy_score(labels_test, predictions))

def findFT(x,y,z):
    FT = []
    for i in range(len(x)):
        if x[i] == 'postive' and y[i] == 'negative':
            FT.append(z[i])
    return FT

def findTF(x,y,z):
    TF = []
    for i in range(len(x)):
        if x[i] == 'negative' and y[i] == 'postive':
            TF.append(z[i])
    return TF
