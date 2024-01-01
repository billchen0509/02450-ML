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
# for nonlinear attributes
ww_idx = attributeNames.index('Whole weight')
sw_idx = attributeNames.index('Shucked weight')
vw_idx = attributeNames.index('Viscera weight')
shw_idx = attributeNames.index('Shell weight')
X[:,ww_idx] = np.power(X[:,ww_idx],2)#.reshape(-1,1)
X[:,sw_idx] = np.power(X[:,sw_idx],2)#.reshape(-1,1)
X[:,vw_idx] = np.power(X[:,vw_idx],2)#.reshape(-1,1)
X[:,shw_idx] = np.power(X[:,shw_idx],2)#.reshape(-1,1)
'''

# ----------add offset values----------
X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
X = np.concatenate((np.ones((N,1)), X), 1)
attributeNames = [u'Offset'] + attributeNames
M = M + 1

# ----------cross validation----------
K = 10
CV = model_selection.KFold(K, shuffle=True)

lambdas = np.power(10., range(-5, 9))

Error_train_reg = np.empty((K, 1))
Error_test_reg = np.empty((K, 1))

Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

Error_train = np.empty((K,1))
Error_test = np.empty((K,1))

#sf = np.asarray([5, 7, 8, 10])
#M = len(sf)
w = np.empty((M, K))
w_reg = np.empty((M, K))
lambdas_l = []
error_i = []

k = 0

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.33, random_state=42)

internal_cross_validation = 10

Error_train_nofeatures = np.sum(np.square(y_train - np.mean(y_train)), axis=0) / np.shape(y_train)[0]
Error_test_nofeatures = np.sum(np.square(y_test - np.mean(y_train)), axis=0) / np.shape(y_test)[0]
w = np.squeeze(np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train))
    
Error_train = np.sum(np.square(y_train - X_train @ w), axis=0) / np.shape(y_train)[0]
Error_test = np.sum(np.square(y_test - X_test @ w), axis=0) / np.shape(y_test)[0]

# -----linear regression with regularization-----
opt_var_err, opt_lambda, mean_w_lambda, train_err_lambda, test_err_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

lambdaI = opt_lambda * np.eye(M)
lambdaI[0,0] = 0
    
w_reg = np.squeeze(np.linalg.solve((X_train.T @ X_train + lambdaI), (X_train.T @ y_train)))
    
Error_train_reg = np.sum(np.square(y_train - X_train @ w_reg), axis=0) / np.shape(y_train)[0]
y_est =  X_test @ w_reg
resi = y_est - y_test
Error_test_reg = np.sum(np.square(y_test - X_test @ w_reg), axis=0) / np.shape(y_test)[0]
    

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
plt.show()

plt.figure(figsize=(12,10))
plt.subplot(2,1,1)
plt.title('Estimated rings vs true rings')
plt.plot(y_test, y_est, '.')
z = np.polyfit(y_test, y_est, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test),'-')
plt.plot(range(1,27), range(1,27), '-')
plt.xlabel('True rings')
plt.ylabel('Estimated rings')
plt.grid()

plt.subplot(2,1,2)
plt.title('Histogram of residuals')
plt.hist(resi, 40, edgecolor='black')
plt.show()

plt.figure(figsize=(15,15))
plt.title('Residuals vs attributes')
for i in range(1, M):
    plt.subplot(2,5,i)
    plt.plot(X_test[:,i], resi, '.')
    plt.xlabel(attributeNames[i])
#plt.suptitle('Abalone dataset - histogram of attributes')
plt.show()

'''
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.semilogx(lambdas, np.mean(lambda_l).T[:, 1:], '.-')
plt.xlabel('Regularization factor')
plt.ylabel('Mean coefficient values')
plt.grid()

plt.subplot(1,2,2)
plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
plt.loglog(lambdas, Error_train_i.T, 'b.-', lambdas, Error_test_i.T, 'r.-')
plt.xlabel('Regularization factor')
plt.ylabel('Cross validation (square error)')
plt.legend(['Train error', 'Test error'])
plt.grid()
plt.show()
'''



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
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_reg[m],4)))
'''
for m, f in enumerate(sf):
    print('{:>15} {:>15}'.format(attributeNames[f], np.round(w_reg[m,-1],2)))
'''
'''
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.plot(range(1, K+1), Error_train, '-')
plt.plot(range(1, K+1), Error_test, '-')
plt.xlabel('K')
plt.ylabel('Generalization Error')
plt.grid()
plt.legend(['Train error', 'Test error'])

plt.subplot(1,2,2)
plt.plot(range(1, K+1), Error_train_reg, '-')
plt.plot(range(1, K+1), Error_test_reg, '-')
plt.xlabel('K')
plt.ylabel('Generalization Error')
plt.legend(['Regularized train error', 'Regularized test error'])
plt.grid()
plt.show()

plt.figure()
plt.plot(lambdas, test_err_lambda, '-')
plt.xlabel('lambdas')
plt.ylabel('Generalization error')
plt.grid()
plt.show()
'''
