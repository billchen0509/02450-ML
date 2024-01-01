# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:44:33 2019

@author: Lychen
"""
import numpy as np
from sklearn import linear_model, model_selection
from toolbox_02450 import train_neural_net, mcnemar
#import matplotlib.pyplot as plt
import torch
from scipy import stats
from prettytable import PrettyTable

# ----------import data----------
infile = (open('abalone.names', 'r')).readlines()
attributeNames = []
line = 89
while line < 97:
    attributeNames.append(((infile[line]).split('\t'))[1])
    line += 1

infile = (open('abalone.data','r')).readlines()
data_r = []
for line in infile:
    data_r += (line.strip()).split(',')

data_r = np.array(data_r).reshape(4177, 9)

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

M = len(attributeNames)  # number of attributes
X = data_r[:,1:].astype(np.float)         # data matix
X = (X - np.ones((len(X), M))*np.mean(X, axis=0))/np.std(X, axis=0)
N = len(data_r[:,0])       # number of observations
C = len(classNames)     # number of class

K=3
CV = model_selection.KFold(K, shuffle=True)

# ----------logr---------
reg_list = [pow(10,i) for i in range(-5,4)]
opt_reg = []
error_lr = []
theta_lr = []

# ----------ANN----------
n_hidden_units_l = [i for i in range(1, 8)]
opt_hidden = []
error_ann = []
n_replicates = 1
max_iter = 10000
theta_ann = []

# ----------basline--------
error_bl = []
theta_bl = []

# ----------performance-----------
alpha = 0.05
CI_data = []
p_data = []
k=0
for (k, (train_index, test_index)) in enumerate(CV.split(X, y)):
    
    # ----------logistic regression----------
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    error_lr_i = np.empty((K, len(reg_list)))
    error_ann_i = np.empty((K, len(n_hidden_units_l)))
    
    for (k_i, (train_index_i, test_index_i)) in enumerate(CV.split(X_train, y_train)):
        X_train_i = X_train[train_index_i,:]
        y_train_i = y_train[train_index_i]
        X_test_i = X_train[test_index_i,:]
        y_test_i = y_train[test_index_i]
        
        #X_train_ann_i = torch.tensor(X_train_i)#, dtype=torch.float)
        #y_train_ann_i = torch.tensor(X)
        
        # lr
        for i, regularization_strength in enumerate(reg_list):
            model_lr = linear_model.LogisticRegression(
                    solver='lbfgs', multi_class='multinomial', 
                    tol=1e-4, random_state=1, 
                    penalty='l2', C=1/regularization_strength, max_iter=10000)    
            y_test_i_est = model_lr.fit(X_train_i, y_train_i).predict(X_test_i)
            error_lr_i[k, i] = np.sum(y_test_i_est!=y_test_i) / len(y_test_i)
        
        # ann
        
        for j, n_hidden_units in enumerate(n_hidden_units_l):
            print('K1: {}, K2: {}, N_hidden_units: {}'.format(k, k_i, n_hidden_units))
            model_ann = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units,C),
                    torch.nn.Softmax(dim=1))
            loss_fn = torch.nn.CrossEntropyLoss()
            net, final_loss, learning_curve = train_neural_net(
                    model_ann, loss_fn, 
                    X=torch.tensor(X_train_i, dtype=torch.float),
                    y=torch.tensor(y_train_i, dtype=torch.long),
                    n_replicates=n_replicates, max_iter=max_iter)
            softmax_logits_i = net(torch.tensor(X_test_i, dtype=torch.float))
            y_test_i_est = (torch.max(softmax_logits_i, dim=1)[1]).data.numpy()
            error_ann_i[k, j] = np.sum(y_test_i_est != y_test_i) / len(y_test_i)
        
    
    # lr
    regularization_strength = reg_list[np.unravel_index(np.argmin(error_lr_i), error_lr_i.shape)[1]]
    model_lr = linear_model.LogisticRegression(
            solver='lbfgs', multi_class='multinomial', 
            tol=1e-4, random_state=1, 
            penalty='l2', C=1/regularization_strength)
    y_test_lr_est = model_lr.fit(X_train, y_train).predict(X_test)
    opt_reg.append(regularization_strength)
    error_lr.append(np.sum(y_test_lr_est != y_test) / len(y_test))
    #theta_lr.append(y_test_lr_est)
    theta_lr = y_test_lr_est
    #theta_lr.append((y_test_lr_est != y_test).astype(int))
    
    # ann
    n_hidden_units = n_hidden_units_l[np.unravel_index(np.argmin(error_ann_i), error_ann_i.shape)[1]]
    model_ann = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, n_hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_units, C),
            torch.nn.Softmax(dim=1))
    loss_fn = torch.nn.CrossEntropyLoss()
    net, final_loss, learning_curve = train_neural_net(
            model_ann, loss_fn,
            X=torch.tensor(X_train, dtype=torch.float),
            y=torch.tensor(y_train, dtype=torch.long),
            n_replicates=n_replicates, max_iter=max_iter)
    softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
    y_test_ann_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
    opt_hidden.append(n_hidden_units)
    error_ann.append(np.sum(y_test_ann_est != y_test) / len(y_test))
    #theta_ann.append((y_test_lr_est != y_test).astype(int))
    #theta_ann.append(y_test_lr_est)
    theta_ann = y_test_lr_est
    
    # bl
    error_bl.append(stats.mode(y_test)[1][0]/len(y_test))
    #theta_bl.append([1 if i==stats.mode(y_test)[0][0] else 0 for i in y_test])
    #theta_bl.append(np.ones((len(y_test),1))*stats.mode(y_test)[0][0])
    theta_bl = np.ones((len(y_test),1))*stats.mode(y_test)[0][0]
    
    k = k + 1

table = PrettyTable()
table.add_column('K', [i for i in range(1,K+1)])
table.add_column('h*', opt_hidden)
table.add_column('E_ann', error_ann)
table.add_column('lambda*', opt_reg)
table.add_column('E_lr', error_lr)
table.add_column('baseline', error_bl)
print(table)

'''
theta_list = [theta_lr, theta_ann, theta_bl]
for i, j in enumerate([1,2,0]):
    [thetahat, CI, p] = mcnemar(np.asarray(theta_true), np.asarray(theta_list[i]), np.asarray(theta_list[j]), alpha=alpha)
    CI_data.append(CI)
    p_data.append(p)
'''

# ----------performance----------

l = opt_reg[error_lr.index(min(error_lr))]
h = opt_hidden[error_ann.index(min(error_ann))]
X_train_p, X_test_p, y_train_p, y_test_p = model_selection.train_test_split(X,y,test_size=0.33, random_state=42)
model_lr = linear_model.LogisticRegression(
        solver='lbfgs', multi_class='multinomial', 
        tol=1e-4, random_state=1, 
        penalty='l2', C=1/l)
y_est_lr = model_lr.fit(X_train_p, y_train_p).predict(X_test_p)

model_ann = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, h),
        torch.nn.ReLU(),
        torch.nn.Linear(h, C),
        torch.nn.Softmax(dim=1))
loss_fn = torch.nn.CrossEntropyLoss()
net, final_loss, learning_curve = train_neural_net(
        model_ann, loss_fn,
        X=torch.tensor(X_train_p, dtype=torch.float),
        y=torch.tensor(y_train_p, dtype=torch.long),
        n_replicates=n_replicates, max_iter=max_iter)
softmax_logits = net(torch.tensor(X_test_p, dtype=torch.float))
y_est_ann = (torch.max(softmax_logits, dim=1)[1]).data.numpy()

y_est_bl = np.ones((len(y_test_p),1))*stats.mode(y_test_p)[0][0]


y_est_list = [y_est_lr, y_est_ann, y_est_bl.reshape(1,len(y_est_bl)).squeeze()]
for i, j in enumerate([1,2,0]):
    [thetahat, CI, p] = mcnemar(y_test_p, y_est_list[i], y_est_list[j], alpha=alpha)
    CI_data.append(CI)
    p_data.append(p)

    
    
    
    
    
    
    
    
