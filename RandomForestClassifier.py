# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:56:16 2020

@author: DGE
"""

import pandas as pd
import numpy as np
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm

#data1
d1 = pd.read_csv("iris.data.multiclass.csv")
X1 = d1.drop(['variety'], axis=1)   #axis1刪除縱列
y1 = d1['variety'].map({'Setosa':1,'Virginica':2,'Versicolor':3})     #y為label欄
#y1 = y1.apply(lambda x:1 if x==0 else -1)
t1 = pd.read_csv("iris.test.multiclass.csv")
#t1.columns = ['a','b','c','d','label']  #test
Xt1 = t1.drop(['variety'], axis=1)
yt1 = t1['variety'].map({'Setosa':1,'Virginica':2,'Versicolor':3})
#yt1 = yt1.apply(lambda x:1 if x==0 else -1)
#X_train, X_test, y_train, y_test = train_test_split(X1.values, y1.values, test_size = 0.20)   #import  8成給X 2成給Y
model = RandomForestClassifier(bootstrap=True, criterion='gini',
            max_depth=10, max_features=3,random_state = 1452,     
            n_estimators=1000)   #X_train.shape[1]=4
model.fit(X1, y1)   #訓練模型

pred = model.predict(Xt1)
print('RandomForestClassifier',accuracy_score(yt1, pred))

clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=1000, 
                       subsample=0.1, criterion='mae', min_samples_split=2, 
                       min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=5, 
                       min_impurity_decrease=0.0, min_impurity_split=None, init=None, 
                       random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, 
                       warm_start=False, presort='auto')
clf.fit(X1, y1)
pred = clf.predict(Xt1)
print('GradientBoostingClassifier',accuracy_score(yt1, pred))

parameters = {
    'application': 'multi-class',
    'objective': 'huber',
    'metric': 'l2',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 10,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}
lgb_train = lightgbm.Dataset(X1, label=y1)
model = lightgbm.train(parameters,
                       lgb_train,
#                        valid_sets=test_data,
                       num_boost_round=1000)

pred = model.predict(Xt1)
pred[pred>0.2]=1
pred[pred<=0.2]=0
pred = np.asarray(pred, np.int)
print('lightgbm',accuracy_score(yt1, pred))

"""y_pred = model.predict(X_test)  #測試結果
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
ans1 = model.predict(Xt1.values)
print(ans1)
print(confusion_matrix(yt1,ans1))
print(classification_report(yt1,ans1))"""
