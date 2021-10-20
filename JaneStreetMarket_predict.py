# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 21:54:38 2020

@author: DGE
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cudf
pd.set_option('display.max_columns', 100)

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import gc

import warnings
warnings.filterwarnings("ignore") #忽略警告訊息

import janestreet
env = janestreet.make_env() # initialize the environment
iter_test = env.iter_test() # an iterator which loops over the test set

import xgboost as xgb
print("XGBoost version:", xgb.__version__) #just run once

def XGB(train_x, train_y):
    model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=11,
    min_child_weight=9.15,
    gamma=0.59,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.7,
    alpha=10.4,
    nthread=5,
    missing=-999,
    random_state=2020,
    tree_method='gpu_hist')  # THE MAGICAL PARAMETER)
    model.fit(train_x, train_y)     # 訓練模型
    return model

def Average(X):
    #feature avg
    num = 0
    for feature in X:
        train[feature] = X[feature].apply(lambda x:1 if x > 0 else -1) #以X為條件將train[feature]的Data分兩類
        check = train[[feature, 'resp']].groupby([feature], as_index=False).mean().sort_values(by='resp', ascending=False) #1,-1分群
        if check['resp'][0] > check['resp'][1]:
            num += check['resp'][0] - check['resp'][1] #求出所有feature差和
        else:
            num += check['resp'][1] - check['resp'][0]
        del check
    avg = num/130
    return avg
    #0.0005447971881263638

def Feature(avg,X):
    val = 0
    i = 0
    for feature in X:
        if (i != 7) and (i != 8 ):
            train[feature] = X[feature].apply(lambda x:1 if x > 0 else -1) #以X為條件將train[feature]的Data分兩類
            check = train[[feature, 'resp']].groupby([feature], as_index=False).mean().sort_values(by='resp', ascending=False) #求兩類平均resp
            if check['resp'][0] > check['resp'][1]: #大-小
                val = check['resp'][0] - check['resp'][1]
            else:
                val = check['resp'][1] - check['resp'][0]
            if (val - avg) > 0:    #差 - 平均差
                X[feature] = X[feature].apply(lambda x:1 if x > 0 else -1)
            del check,val
        i+=1
    del i
    return X

%%time
#load Data
train_cudf  = cudf.read_csv('/kaggle/input/jane-street-market-prediction/train.csv')
train = train_cudf.to_pandas()
del train_cudf
features = pd.read_csv('../input/jane-street-market-prediction/features.csv')
example_test = pd.read_csv('../input/jane-street-market-prediction/example_test.csv')
sample_prediction_df = pd.read_csv('../input/jane-street-market-prediction/example_sample_submission.csv')
print ("Data is loaded!")

#draw line chart
fig, ax = plt.subplots(figsize=(15, 5))
balance= pd.Series(train['resp']).cumsum()
resp_1= pd.Series(train['resp_1']).cumsum()
resp_2= pd.Series(train['resp_2']).cumsum()
resp_3= pd.Series(train['resp_3']).cumsum()
resp_4= pd.Series(train['resp_4']).cumsum()
ax.set_xlabel ("Trade", fontsize=18)
ax.set_title ("Cumulative resp and time horizons 1, 2, 3, 4", fontsize=18)
balance.plot(lw=3)
resp_1.plot(lw=3)
resp_2.plot(lw=3)
resp_3.plot(lw=3)
resp_4.plot(lw=3)
plt.legend(loc="upper left");
del resp_1
del resp_2
del resp_3
del resp_4
gc.collect();

#draw Histogram
plt.figure(figsize = (12,5))
ax = sns.distplot(train['feature_8'], 
             bins=3000, 
             kde_kws={"clip":(-2,2)}, 
             hist_kws={"range":(-2,2)},
             color='darkcyan', 
             kde=False);
values = np.array([rec.get_height() for rec in ax.patches])
norm = plt.Normalize(values.min(), values.max())
colors = plt.cm.jet(norm(values))
for rec, col in zip(ax.patches, colors):
    rec.set_color(col)
plt.xlabel("Histogram of the resp values", size=14)
plt.show();
del values
gc.collect();

#draw missing value 
miss = train.isnull().sum()
px.bar(miss, color=miss.values, title="Total number of missing values for each column").show()
del miss

#missing value%
missing_values_count = train.isnull().sum()
print (missing_values_count)
total_cells = np.product(train.shape)
total_missing = missing_values_count.sum()
m = missing_values_count['feature_114'].sum()
c = np.product(train['feature_114'].shape)
print("features % = ",m/c*100)
print ("% of missing data = ",(total_missing/total_cells) * 100)
del m,c

#check
check = train[['feature_7', 'resp']].groupby(['feature_7'], as_index=False).mean().sort_values(by='resp', ascending=False)
ck = train['feature_8'].mean()
print(ck)
del check,ck

#trainX,labelY
target = train['weight'] != 0 #weight==0無參考價值
train = train[target]
train['action'] = (train['resp'].values > 0).astype('int') #>0才進場投資
X = train.loc[:, train.columns.str.contains('feature')]  #取出有features的
#X = X.drop(['feature_17','feature_18','feature_27','feature_28','feature_84','feature_90','feature_96','feature_102','feature_108','feature_114'],axis=1)
y = train['action']
del target

%%time
#fillna
col = {'feature_7': np.product(X['feature_7'].mode()),'feature_8': np.product(X['feature_8'].mode()),'feature_72': np.product(X['feature_72'].mode()),'feature_78': np.product(X['feature_78'].mode())}
X = X.fillna(value=col)
X = X.fillna(X.mean())
del col
print(X)

%%time
avg = Average(X)
print(avg)
#0.0005447971881263638

%%time
X = Feature(avg,X)
del train

print(X)

%%time
model = XGB(X,y)

%%time
for (test_df, sample_prediction_df) in iter_test:
    X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
    y_preds = model.predict(X_test)
    sample_prediction_df.action = y_preds
    env.predict(sample_prediction_df)
