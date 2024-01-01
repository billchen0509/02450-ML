# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:44:33 2019

@author: Lychen
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate, feature_selector_lr, bmplot

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

data[:,3:-1] = data_r[:,1:-1].astype(np.float)
data[:,-1] = data_r[:,-1].astype(np.int)

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

lambdas = np.power(10., range(-5, 9))

Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_reg = np.empty((K, 1))
Error_test_reg = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
Error_train_fs = np.empty((K, 1))
Error_test_fs = np.empty((K, 1))
Features = np.zeros((M, K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
#sf = np.asarray([5, 7, 8, 10])
#M = len(sf)
w = np.empty((M, K))
w_reg = np.empty((M, K))
lambdas_l = []

k = 0
for train_index, test_index in CV.split(X, y):
    
    X_train = X[train_index]
    X_train_fs = X_train[:, 1:]
    X_test = X[test_index]
    X_test_fs = X_test[:, 1:]
    y_train = y[train_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    mu[k, :] = np.mean(X_train[:, 1:], axis=0)
    sigma[k, :] = np.std(X_train[:, 1:], axis=0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]
    
    Error_train_nofeatures[k] = np.sum(np.square(y_train - np.mean(y_train)), axis=0) / np.shape(y_train)[0]
    Error_test_nofeatures[k] = np.sum(np.square(y_test - np.mean(y_train)), axis=0) / np.shape(y_test)[0]
    '''
    # -----feature selection-----
    textout = ''
    selected_features, features_record, loss_record = feature_selector_lr(X_train_fs, y_train, internal_cross_validation, display=textout)
    Features[selected_features,k]=1
    if len(selected_features) == 0:
        print('No features were selected')
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train_fs[:,selected_features], y_train)
        Error_train_fs[k] = np.sum(np.square(y_train-m.predict(X_train_fs[:,selected_features])))/np.shape(y_train)[0]
        Error_test_fs[k]= np.sum(np.square(y_test-m.predict(X_test_fs[:,selected_features])))/np.shape(y_test)[0]
        
        plt.figure(k, figsize=(10, 10))
        plt.subplot(1,2,1)
        plt.plot(range(1, len(loss_record)), loss_record[1:])
        plt.xlabel('Iteration')
        plt.ylabel('Cross vadiation (squared error)')
        
        plt.subplot(1,3,3)
        bmplot(attributeNames[1:], range(1, features_record.shape[1]), -features_record[:,1:])
        plt.clim(-1.5,0)
        plt.xlabel('Iteration')
    '''
    #X_train = X_train[:, sf]
    #X_test = X_test[:, sf]
    
    # -----linear regression without regularization-----
    w[:, k] = np.squeeze(np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train))
    
    Error_train[k] = np.sum(np.square(y_train - X_train @ w[:,k]), axis=0) / np.shape(y_train)[0]
    Error_test[k] = np.sum(np.square(y_test - X_test @ w[:, k]), axis=0) / np.shape(y_test)[0]
    
    # -----linear regression with regularization-----
    opt_var_err, opt_lambda, mean_w_lambda, train_err_lambda, test_err_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    lambdas_l.append(opt_lambda)
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0
    
    w_reg[:, k] = np.squeeze(np.linalg.solve((X_train.T @ X_train + lambdaI), (X_train.T @ y_train)))
    
    Error_train_reg[k] = np.sum(np.square(y_train - X_train @ w_reg[:, k]), axis=0) / np.shape(y_train)[0]
    Error_test_reg[k] = np.sum(np.square(y_test - X_test @ w_reg[:, k]), axis=0) / np.shape(y_test)[0]
    
    if k == K-1:
        plt.figure(figsize=(15,10))
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
       
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_reg.mean()))
print('- Test error:     {0}'.format(Error_test_reg.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_reg.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_reg.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')

for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_reg[m,-1],2)))
'''
for m, f in enumerate(sf):
    print('{:>15} {:>15}'.format(attributeNames[f], np.round(w_reg[m,-1],2)))
'''

plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.plot(range(1, K+1), Error_train, '-')
plt.plot(range(1, K+1), Error_test, '-')
plt.xlabel('K')
plt.ylabel('Error')
plt.grid()
plt.legend(['Error train', 'Error test'])

plt.subplot(1,2,2)
plt.plot(range(1, K+1), Error_train_reg, '-')
plt.plot(range(1, K+1), Error_test_reg, '-')
plt.xlabel('K')
plt.ylabel('Error')
plt.legend(['Error train reg', 'Error test reg'])
plt.grid()
plt.show()
