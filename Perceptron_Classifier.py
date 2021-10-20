# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 20:56:16 2020

@author: DGE
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn import preprocessing

class Perceptron(object):

    def __init__(self, no_of_inputs, iterations=100, learning_rate=0.001):   #在class中直接執行
        self.iterations = iterations
        self.lr = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)   #np.zero(shape)        
           
    def predict(self, inputs):
        results = []
        for item in inputs:
            pred = np.dot(item, self.weights[1:]) + self.weights[0] #x1*w1+x2*w2+...xn*wn + b(bias)
            results.append(np.sign(pred))
        return results

    def train(self, training_inputs, labels):
        for _ in range(self.iterations):
            for inputs, label in zip(training_inputs, labels):    #zip矩陣轉換
                prediction = self.predict([inputs])
                self.weights[1:] += self.lr * (label - prediction) * inputs
                self.weights[0] +=  (label - prediction)

def convert(X,item):
    return X.index(item)

#normalized = True

heart = pd.read_csv("crx.csv")
#print(heart.head())     #前五筆
#print(heart.info())     #找出資料型態
X = heart.drop(['att1','label'], axis=1)   #axis1刪除縱col
feat = ['att2','att4','att5','att6','att7','att9','att10','att11','att12','att13','att14','att15']

for key in feat:
    sets = list(np.unique(X[key].values))   #feat為list型態
    X[key] = X[key].apply(lambda x:convert(sets, x))    #依序傳入X[key]的每一項，reutrn回來的list

y = heart['label']      #y為label欄
y = y.apply(lambda x: 1 if x == '+' else -1)   #符合條件為1不符為-1
#print(X)

#X = X.values
#print(X)
"""min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(X_scaled)"""
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size = 0.20)   #隨機8成給X 2成給Y

#print(X_train)
#print(X_test)
#print(np.zeros(X_train.shape[1]))

perceptron = Perceptron(X_train.shape[1])   #X_train.shape[1]=14
perceptron.train(X_train, y_train)
y_pred = perceptron.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
