# # -*- coding: utf-8 -*-
# """
# Spyder Editor

# This is a temporary script file.
# """
import pandas as pd
import math
import copy

test = pd.read_csv('20201005-weather.csv')
X = test.iloc[:, :].values #[row,col]
attribute = ['Hour','Weather', 'Accident', 'Stall']
#print(X)
#print(len(X[0])-1)

class Node(object):
    def __init__(self):
        self.value = None
        self.decision = None
        self.childs = None

def findEntropy(data, rows):
    long = 0
    short = 0
    ans = -1
    idx = len(data[0]) - 1      #col數量(-1為0開始)
    entropy = 0     #接收的每條訊息中包含的資訊的平均量(ML)
    for i in rows:
        if data[i][idx] == 'Long':
            long = long + 1
        else:
            short = short + 1

    x = long / (long+short)
    y = short / (long+short)
    if x != 0 and y != 0:
        entropy = -1 * (x*math.log2(x) + y*math.log2(y))    #C4.5算法
    if x == 1:
        ans = 1
    if y == 1:
        ans = 0
    return entropy, ans


def findMaxGain(data, rows, columns):
    maxGain = 0
    retidx = -1
    entropy, ans = findEntropy(data, rows)
    if entropy == 0:
        """if ans == 1:
            print("Yes")
        else:
            print("No")"""
        return maxGain, retidx, ans

    for j in columns:
        mydict = {}
        idx = j
        for i in rows:      #分類
            key = data[i][idx]
            if key not in mydict:
                mydict[key] = 1
            else:
                mydict[key] = mydict[key] + 1
        gain = entropy

        # print(mydict)
        for key in mydict:  #計算col不同類別的gain
            long = 0
            short = 0
            for k in rows:
                if data[k][j] == key:
                    if data[k][-1] == 'Long':
                        long = long + 1
                    else:
                        short = short + 1
            # print(yes, no)
            x = long/(long+short)       #計算columns的Long機率
            y = short/(long+short)
            # print(x, y)
            if x != 0 and y != 0:
                gain += (mydict[key] * (x*math.log2(x) + y*math.log2(y)))/12
        # print(gain)
        if gain > maxGain:
            # print("hello")
            maxGain = gain
            retidx = j

    return maxGain, retidx, ans


def buildTree(data, rows, columns):
    maxGain, idx, ans = findMaxGain(X, rows, columns)
    root = Node()
    root.childs = []
    # print(maxGain
    #
    # )
    if maxGain == 0:
        if ans == 1:
            root.value = 'Long'
        else:
            root.value = 'Short'
        return root

    root.value = attribute[idx]     # columns's Title
    mydict = {}
    for i in rows:
        key = data[i][idx]
        if key not in mydict:
            mydict[key] = 1
        else:
            mydict[key] += 1

    newcolumns = copy.deepcopy(columns)
    newcolumns.remove(idx)
    for key in mydict:
        newrows = []
        for i in rows:
            if data[i][idx] == key:
                newrows.append(i)
        # print(newrows)
        temp = buildTree(data, newrows, newcolumns)
        temp.decision = key
        root.childs.append(temp)
    return root

def traverse(root):
    print(root.decision)
    print(root.value)

    n = len(root.childs)
    if n > 0:
        for i in range(0, n):
            traverse(root.childs[i])

def calculate():
    rows = [i for i in range(0, len(X))]
    columns = [i for i in range(0, len(X[0])-1)]
    root = buildTree(X, rows, columns)
    root.decision = 'Start'
    traverse(root)
    
calculate()
