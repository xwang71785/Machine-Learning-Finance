# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:57:01 2017
xgboost
@author: wangx3
"""

import numpy as np
import xgboost as xgb

from sklearn.datasets import load_svmlight_files
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier

datatrain = 'data/agaricus.txt.train'
datatest = 'data/agaricus.txt.test'

# sklearn interface
X_train, y_train, X_test, y_test = load_svmlight_files((datatrain, datatest))
params = {'objective':'binary:logistic', 
          'max_depth': 3, 
          'silent': 1.0, 
          'learning_rate': 0.1,
          'n_estimators': 10}

bst = XGBClassifier(**params).fit(X_train, y_train)
preds = bst.predict(X_test)
acc = accuracy_score(y_test, preds)
print(acc)

'''
# native interface

dtrain = xgb.DMatrix(datatrain)
dtest = xgb.DMatrix(datatest)

params = {'objective':'binary:logistic', 
          'max_depth':2, 
          'silent':1, 
          'eta':1}    # learning rate
num_rounds = 5

bst = xgb.train(params, dtrain, num_rounds)

preds_prob = bst.predict(dtest)
'''








