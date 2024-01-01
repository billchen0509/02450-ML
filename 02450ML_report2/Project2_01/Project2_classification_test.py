# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:44:33 2019

@author: Lychen
"""
import numpy as np
from sklearn import linear_model, model_selection
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
import matplotlib.pyplot as plt

# ----------import data----------
infile = (open('abalone.names', 'r')).readlines()
attributeNames = []
line = 88
while line < 97:
    attributeNames.append(((infile[line]).split('\t'))[1])
    line += 1

infile = (open('abalone.data','r')).readlines()
data_r = []
for line in infile:
    data_r += (line.strip()).split(',')

data_r = np.array(data_r).reshape(4177, 9)


'''
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

data[:,3:-1] = data_r[:,1:-1].astype(np.float)
data[:,-1] = data_r[:,-1].astype(np.int)

data = np.delete(data, data[:,2].argmax(axis=0), axis=0)
'''
classLabels = (data_r[:,0]).tolist()

'''
# regression
y = np.asarray(classLabels[:]).reshape(len(classLabels),1)
'''
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
'''
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(len(classNames))))

y = np.asarray([classDict[value] for value in classLabels])    # class index

attributeNames_n = attributeNames[4:10:4]   # [4:10:4] & [5:10:3]

X = data_r[:,4:10:4].astype(np.float)         # data matix
N = len(data_r[:,0])       # number of observations
M = len(attributeNames)  # number of attributes
C = len(classLabels)     # number of class

K=3
CV = model_selection.KFold(K, shuffle=True)

regularization_strength = 1e-3

for (k, (train_index, test_index)) in enumerate(CV.split(X, y)):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    model = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, random_state=1, penalty='l2', C=1/regularization_strength)    
    model.fit(X_train, y_train)
    
    y_test_est = model.predict(X_test)
    
    test_error_rate = np.sum(y_test != y_test_est) / len(y_test)
    
    print('Number of miss-classifications for multinomial regression: \n\t {} out of {}'.format(test_error_rate*len(y_test), len(y_test)))
    
    
    predict = lambda x: np.argmax(model.predict_proba(x), 1)
    plt.figure(2, figsize=(9,9))
    visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames_n, classNames)
    plt.title('Logistic regression decision boundaries')
    plt.show()
    
    plt.figure(2, figsize=(9,9))
    plt.hist([y_train, y_test, y_test_est], color=['red','green','blue'], density=True)
    plt.legend(['Training labels','Test labels','Estimated test labels'])
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
