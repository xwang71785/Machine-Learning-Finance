# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:25:33 2016
Kaggle Enhance
@author: wangx3
"""

measurements = [{'city':'dubai', 'temperature':33.}, 
                {'city':'London', 'temperature':12.}, 
                {'city':'San Fransisco', 'temperature':18.}]

from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()
print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
news = fetch_20newsgroups(subset='all')
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target,
                                                    test_size=0.25, random_state=33)
count_vec = CountVectorizer()
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)
























