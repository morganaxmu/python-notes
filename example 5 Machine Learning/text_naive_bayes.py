# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:05:02 2020

@author: billy huang
"""

#Text classification with Naive Bayes


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

documents = ['Thanks for booking your place for the session today at 4pm',
             'Look forward to seeing you all later',
             "If you have any issues please contact",
             "If you ever want to remove yourself from this mailing list",
             "you can send mail to  with the following message"]


vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
tfidf = vectorizer.fit_transform(documents)
pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
print(vectorizer.idf_)
pd.DataFrame([vectorizer.idf_], columns=vectorizer.get_feature_names())
idf_data = pd.DataFrame([vectorizer.idf_], columns=vectorizer.get_feature_names()).T
print(idf_data.sort_values(by=0, axis=0))
