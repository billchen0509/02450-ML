# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:44:33 2019

@author: Lychen
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection

# ----------import data----------
infile = (open('abalone.names', 'r')).readlines()
attributeNames = []
line = 88
while line < 96:
    attributeNames.append(((infile[line]).split('\t'))[1])
    line += 1

infile = (open('abalone.data','r')).readlines()
data_r = []
for line in infile:
    data_r += (line.strip()).split(',')

data_r = np.array(data_r).reshape(4177, 9)
data = np.zeros((4177,11))

attributeNames[0] = 'M'
attributeNames.insert(0, 'I')
attributeNames.insert(0, 'F')

for i in range(len(data_r[:,0])):
    if data_r[i,0] == 'F':
        data[i,0] = 1
    elif data_r[i,0] == 'I':
        data[i,1] = 1
    elif data_r[i,0] == 'M':
        data[i,2] = 1    

data[:,3:-1] = data_r[:,1:-1]
data[:,-1] = data_r[:,-1]

data = np.delete(data, data[:,2].argmax(axis=0), axis=0)

classLabels = (data[:,-1]).tolist()

# regression
y = np.asarray(classLabels[:])

'''
# classification
for i in range(len(classLabels)):
    num = int(classLabels[i])
    if num in range(1, 6):
        classLabels[i] = '1'
    if num in range(6, 14):
        classLabels[i] = '2'
    if num in range(14, 31):
        classLabels[i] = '3'

classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classNames))))

y = np.asarray([classDict[value] for value in classLabels])    # class index
'''
X = data[:,0:-1]         # data matix
N = len(data[:,0])       # number of observations
M = len(attributeNames)  # number of attributes
C = len(classLabels)     # number of class

'''
# ----------boxplot----------
plt.figure(figsize=(15,10))
plt.boxplot(X)
plt.xticks(range(1, M+1), attributeNames, fontsize=10)
plt.ylabel('mm or grams', fontsize=15)
plt.title('Abalone dataset - boxplot', fontsize=15)
plt.show()
'''

# ----------add offset values----------
X = np.concatenate((np.ones((N,1)), X), 1)
attributeNames = [u'Offset'] + attributeNames
M = M + 1

# ----------cross validation----------
K = 10
CV = model_selection.KFold(K, shuffle=True)

baseline = []

k = 0
for train_index, test_index in CV.split(X, y):
    
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    #internal_cross_validation = 10
    
    baseline.append(np.sum(y_train)/len(y_train))
    
    k = k + 1

print('Baseline of {} fold:\n{}'.format(K, baseline))

plt.figure(figsize=(15,10))
plt.plot(range(1, K+1), baseline, '-')
plt.xlabel('K-fold')
plt.ylabel('Baseline')
plt.grid()
plt.show()