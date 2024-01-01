# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:44:33 2019

@author: Lychen
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

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
K = 5
CV = model_selection.KFold(K, shuffle=True)

lambdas = np.power(10., range(-5, 9))

Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_reg = np.empty((K, 1))
Error_test_reg = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w = np.empty((M, K))
w_reg = np.empty((M, K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))


k = 0
for train_index, test_index in CV.split(X, y):
    
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    opt_var_err, opt_lambda, mean_w_lambda, train_err_lambda, test_err_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    
    mu[k, :] = np.mean(X_train[:, 1:], axis=0)
    sigma[k, :] = np.std(X_train[:, 1:], axis=0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]
    
    Error_train_nofeatures[k] = np.sum(np.square(y_train - np.mean(y_train)), axis=0) / np.shape(y_train)[0]
    Error_test_nofeatures[k] = np.sum(np.square(y_test - np.mean(y_train)), axis=0) / np.shape(y_test)[0]
    
    # -----linear regression without regularization-----
    w[:, k] = np.squeeze(np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train))
    
    Error_train[k] = np.sum(np.square(y_train - X_train @ w[:,k]), axis=0) / np.shape(y_train)[0]
    Error_test[k] = np.sum(np.square(y_test - X_test @ w[:, k]), axis=0) / np.shape(y_test)[0]
    
    # -----linear regression with regularization-----
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0
    
    w_reg[:, k] = np.squeeze(np.linalg.solve((X_train.T @ X_train + lambdaI), (X_train.T @ y_train)))
    
    Error_train_reg[k] = np.sum(np.square(y_train - X_train @ w_reg[:, k]), axis=0) / np.shape(y_train)[0]
    Error_test_reg[k] = np.sum(np.square(y_test - X_test @ w_reg[:, k]), axis=0) / np.shape(y_test)[0]
    
    if k == K-1:
        plt.figure(figsize=(12,8))
        plt.subplot(1,2,1)
        plt.semilogx(lambdas, mean_w_lambda.T[:, 1:], '.-')
        plt.xlabel('Regularization factor')
        plt.ylabel('Mean coefficient values')
        plt.grid()
        
        plt.subplot(1,2,2)
        plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        plt.loglog(lambdas, train_err_lambda.T, 'b.-', lambdas, test_err_lambda.T, 'r.-')
        plt.xlabel('Regularization factor')
        plt.ylabel('Cross validation (square error)')
        plt.legend(['Train error', 'Test error'])
        plt.grid()
    
    k = k + 1

plt.show()
       

# ----------regularization----------
'''
w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

y_est1 = np.dot(X, w)
'''

'''
ww_idx = attributeNames.index('Whole weight')
ww = np.power(X[:, ww_idx], 2).reshape(-1, 1)
skw_idx = attributeNames.index('Shucked weight')
skw = np.power(X[:, skw_idx], 2).reshape(-1, 1)
vw_idx = attributeNames.index('Viscera weight')
vw = np.power(X[:, vw_idx], 2).reshape(-1, 1)
slw_idx = attributeNames.index('Shell weight')
slw = np.power(X[:, slw_idx], 2).reshape(-1, 1)
X = np.asarray(np.bmat('X, ww, skw, vw, slw'))
'''

'''
ww_idx = attributeNames.index('Whole weight')
X[:, ww_idx] = np.power(X[:, ww_idx], 2)
skw_idx = attributeNames.index('Shucked weight')
X[:, skw_idx] = np.power(X[:, skw_idx], 2)
vw_idx = attributeNames.index('Viscera weight')
X[:, vw_idx] = np.power(X[:, vw_idx], 2)
slw_idx = attributeNames.index('Shell weight')
X[:, slw_idx] = np.power(X[:, slw_idx], 2)
''' 

'''
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(X,y)

y_est = model.predict(X)

residual = y_est - y

plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
plt.plot(y, y_est, '.')
plt.xlabel('Age (real)')
plt.ylabel('Age (estimated)')
plt.subplot(1, 2, 2)
plt.hist(residual, 40)
plt.title('Histogram of residuals')
plt.show()
'''

'''
plt.figure(figsize=(15,10))
plt.plot(X[:,6], y, 'o')
plt.plot(X[:,6], y_est, '.')
plt.xlabel('X')
plt.ylabel('y')
plt.legend(['Training data', 'Regression model'])
'''
