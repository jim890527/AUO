# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:29:30 2020

@author: DGE
"""
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import svm

X = np.array([[-2,-4],[-4,1],[1, 6],[2, 4],[-6, -2]])  #data

y = np.array([-1,-1,1,1,-1])    #label

w = [0,1]   #垂直Y軸且過原點的直線

W=[]

wnorm=[]    #w範圍

u=0

def perceptron(X, Y):
    w = [0.1,0.1]
    eta = 0.001
    epochs = 100
    for t in range(epochs):
        err=0.
        for i, x in enumerate(X):   #i=0~4
            if (np.dot(X[i], w)*Y[i]) < 1:  #<1代表有誤
                w = w + eta*X[i]*Y[i]      #改變w
                W.append(w)     #將w新增至W[list]的最後
                wnorm.append(w[0]**2+w[1]**2)
                err+=1      #計算錯誤率
#         print("#err=%f"%(err/5))
    return w

def margin(W,X):    #最大margin
    w = (W[0]**2 + W[1]**2)**0.5
    print(w)
    maxd = 0
    for x,y in X:
        d = (W[0]*x + W[1]*y) /w
        if(maxd < d):
            maxd = d
    print(maxd)

def Margin(wnorm):  #最適w
    optimal_w = []  #最佳解
    wnorm = np.asarray(wnorm)   #將矩陣轉換為陣列
    idx = wnorm.argsort()   #順序(小到大
    for i in range(len(W)):     #perceptron後的每個W都執行
        ispass = True
        w = W[idx[i]]   
        margin = 0
        for j, x in enumerate(X):   #每個點
            pred_res = np.dot(X[j], w)*y[j]
            #print(j,x)
            if (pred_res<1):    #任一點無法區別都不可
                ispass = False
            margin += (np.dot(X[j], w) /np.prod(w) *y[j])
            #print(margin)
        if ispass:
            optimal_w = w   #找到最適當的線  
    print('margin=',margin)
    return optimal_w

def adline(w, intercept):
    """Plot a line from slope and intercept"""
    if w[0] == 0:
        w[0] = 1e-9
    slope = -w[1]/(w[0])
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    #origin=[np.mean(x_vals), np.mean(y_vals)]
    plt.text(0,0, 'W=' + str(w))
    plt.plot(x_vals, y_vals, '--')

def draw(optimal_w):
    if optimal_w == []:
        print('Not found!!')
    else:
        clf = svm.SVC(kernel ="linear")
        clf.fit(X, y)
        plt.scatter(X[y==-1,0], X[y==-1,1],color='red')
        plt.scatter(X[y==1,0], X[y==1,1],color='blue')
        adline(optimal_w, 1)
        plt.show()

def update(x,y):
    global w,u
    if (y<0):   #y-
        if (w[0]*x[0] + w[1]*x[1] > 0):
            w[0]+=1     #(讓W漸漸為負斜率)
            #w[1]+=1
            u = 0
            #print(w)
            update(x,y)
    else:   #y+
        if (w[0]*x[0] + w[1]*x[1] < 0):
            w[0]-=1     #(讓W漸漸為正斜率)
            #w[1]-=1
            u = 0
            #print(w)
            update(x,y)

#Main
while(u==0):    #若改變W則從頭做
    u = 1
    for i in range(0, len(X)):
        if u==0:
            break
        print(i)
        update(X[i],y[i])
print(w)    
w = perceptron(X,y)
#print(W)
w = Margin(wnorm)
print(w)
draw(w)
