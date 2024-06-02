# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 21:19:19 2020

@author: DGE
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def RandomForestClassifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(train_x, train_y)     # 訓練模型
    return model

#data feature
def fixD(df):
    df['4'] = df['4'].map({'A': 1 , 'B':2,'C':3}).astype(int)
    title = df['6'].drop_duplicates(keep='first')   #取出所有精液編號
    class_mapping = {label: idx for idx, label in enumerate(title)}
    df['6'] = df['6'].map(class_mapping)
    title = df['7'].drop_duplicates(keep='first')
    class_mapping = {label: idx for idx, label in enumerate(title)}
    df['7'] = df['7'].map(class_mapping)
    # title = df['17'].drop_duplicates(keep='first')
    # class_mapping = {label: idx for idx, label in enumerate(title)}
    # df['17'] = df['17'].map(class_mapping)
    # title = df['21'].drop_duplicates(keep='first')
    # class_mapping = {label: idx for idx, label in enumerate(title)}
    # df['21'] = df['21'].map(class_mapping)
    df['12'] = pd.to_datetime(df['12']).dt.month    #.year month 
    df['13'] = pd.to_datetime(df['12']).dt.month
    df['15'] = pd.to_datetime(df['15']).dt.month
    df['16'] = pd.to_datetime(df['16']).dt.month
    df['19'] = pd.to_datetime(df['19']).dt.month
    df['20'] = pd.to_datetime(df['20']).dt.month
    return df

def com(train, test):
    combine = [train, test]
    for dataset in combine:     #train test data
        dataset.loc[(dataset['3'] >= 2) & (dataset['3'] <= 4), 'Season'] = '1'
        dataset.loc[(dataset['3'] == 1) | (dataset['3'] == 5) | (dataset['3'] == 6), 'Season'] = '2'
        dataset.loc[(dataset['3'] >= 10) & (dataset['3'] <= 12) | (dataset['3'] == 7), 'Season'] = '3'
        dataset.loc[(dataset['3'] >= 8) & (dataset['3'] <= 9), 'Season'] = '4'       
        dataset.loc[(dataset['9'] >= 2) & (dataset['9'] <= 4), 'Parity'] = '1'
        dataset.loc[(dataset['9'] == 1) | (dataset['9'] == 6) | (dataset['9'] == 5), 'Parity'] = '2'
        dataset.loc[(dataset['9'] == 11) | (dataset['9'] == 7) | (dataset['9'] == 8), 'Parity'] = '3'
        dataset.loc[(dataset['9'] ==  9), 'Parity'] = '4'
        dataset.loc[(dataset['9'] ==  0), 'Parity'] = '5'
        dataset.loc[(dataset['14'] >=  1) & (dataset['14'] <= 10), 'age'] = '1'
        dataset.loc[(dataset['14'] >= 21) & (dataset['14'] <= 50), 'age'] = '2'     #24
        dataset.loc[(dataset['14'] >= 51) & (dataset['14'] <= 70), 'age'] = '3'     #23.8
        dataset.loc[(dataset['14'] >= 71) & (dataset['14'] <= 80), 'age'] = '4'     #22.8
        dataset.loc[(dataset['14'] >=111) & (dataset['14'] <=120), 'age'] = '4'     #22.9
        dataset.loc[(dataset['14'] >= 81) & (dataset['14'] <= 90), 'age'] = '5'     #21.8
        dataset.loc[(dataset['14'] >= 11) & (dataset['14'] <= 20), 'age'] = '6'     #20.1
        dataset.loc[(dataset['14'] >= 91) & (dataset['14'] <=100), 'age'] = '6'
        dataset.loc[(dataset['14'] >=101) & (dataset['14'] <=110), 'age'] = '7'     #20
        dataset.loc[(dataset['14'] >=121) & (dataset['14'] <=140), 'age'] = '7'
        dataset.loc[(dataset['14'] >=141) & (dataset['14'] <=150), 'age'] = '8'
        dataset.loc[(dataset['14'] >=151) & (dataset['14'] <=160) | (dataset['14'] == 0), 'age'] = '9'     #16.7
        #配種次數
        dataset.loc[(dataset['18'] >=  0) & (dataset['18'] <= 2), 'breedingNo'] = '1'
        dataset.loc[(dataset['18'] >=  3) & (dataset['18'] <= 5), 'breedingNo'] = '2'
        dataset.loc[(dataset['18'] >=  6) & (dataset['18'] <= 8) | (dataset['18'] == 14), 'breedingNo'] = '3'
        dataset.loc[(dataset['18'] >=  9) & (dataset['18'] <= 15), 'breedingNo'] = '4'
        dataset.loc[(dataset['18'] == 16), 'breedingNo'] = '5'
    return train,test

#Main         
re = pd.read_csv("report.csv")     #nrows=1000讀取X行
br = pd.read_csv("birth.csv")
sp = pd.read_csv("spec.csv")
br = pd.read_csv("breed.csv")

re['11'] = re['11'].fillna(-1)  #false
t = np.copy(np.array(re['11']))
test = t == -1  #address
train = test == False   #address
train = re[train]
test = re[test]
train = train.fillna(0)
test = test.fillna(0)
train = fixD(train)
test = fixD(test)

#check
ttt = train[['12', '11']].groupby(['12'], as_index=False).mean().sort_values(by='11', ascending=False)  
print(ttt)
#a = train['10'] == 43
#print((train[a]))
print(train['10'].mean())

#data feature
train,test = com(train,test) 
train = train.drop(['2','3','8','9','14','17','18','21'], axis=1)
test = test.drop(['2','3','8','9','14','17','18','21'], axis=1)
print(train)

#train
X = train.drop(['11'], axis=1)   #axis1刪除縱col
y = train['11'].astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size = 0.2)   #8成Train 2成Test
# #print(X_train.size)

model = RandomForestClassifier(X, y)
train = model.predict(X)    # 訓練所預測出的結果
#print(y_test)   # train的真實結果

#print(confusion_matrix(y,train))       #計算模組的準確率
#print(classification_report(y,train))      #計算模組的準確率

#test
te = test.drop(['11'], axis=1)
ans = model.predict(te)     # 丟進模組預測結果
print('ans=',ans)

ID = te['1']
#print(No)

data={'ID':ID,'milk':ans}
#print(data)

df = pd.DataFrame(data)
#print(df)

df.to_csv('submission.csv',index=0)

#https://www.itread01.com/content/1546068790.html
