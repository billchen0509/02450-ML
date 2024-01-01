#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:36:24 2023

@author: billhikari
"""
# import packages
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid,plot,hist)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from toolbox_02450 import rlr_validate,train_neural_net, draw_neural_net,visualize_decision_boundary
from scipy import stats
import torch
from prettytable import PrettyTable
# load data
path = '/Users/billhikari/Documents/02450 ML/02450ML_report1/forest_fire.csv'

df = pd.read_csv(path, header = 1)

df = df.drop(index = [122,123,124])

df = df.drop(columns = ['day','month','year'])

df.columns.values[0] = 'Temp'

raw_data = df.values

cols = range(0,10)
X = raw_data[:, cols]

# convert X string array to float array
X = np.asfarray(X,dtype = float)

# extract the attribute names 
attributeNames = np.asarray(df.columns[cols])

# extract the last column
classLabels = raw_data[:,-1]

df.iloc[:,-1] = df.iloc[:,-1].apply(lambda x: x.strip())

#determine the class labels
classNames = np.flip(np.unique(classLabels))

classDict = dict(zip(classNames,range(len(classNames))))

# Fire or not
y = np.array([classDict[cl] for cl in classLabels])

C = len(classNames)

N,M = X.shape

# Normalize data
X = stats.zscore(X);
#%%
K = 10
CV = model_selection.KFold(K,shuffle=True,random_state=42)

# logistic regression
reg_list = [pow(10,i) for i in range(-5,5)]
opt_reg = []
error_lr = []

# ANN
max_iter = 10000
n_hidden_units_l = [1,2,3,4,5]
n_replicates = 1
error_ann = []
opt_hidden_units = []

# baseline
error_bl = []

# Outer cross validation loop
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
    # linear regression
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10 
    
    error_ann_i = np.empty((K, len(n_hidden_units_l)))
    error_lr_i = np.empty((K, len(reg_list)))
    
    
    for (j, (train_index_inner, test_index_inner)) in enumerate(CV.split(X_train,y_train)):
        X_train_inner = X_train[train_index_inner,:]
        y_train_inner = y_train[train_index_inner]
        X_test_inner = X_train[test_index_inner,:]
        y_test_inner = y_train[test_index_inner]
        
        #logistic parameter tuning
        for regularization in enumerate(reg_list):
            model_lr = LogisticRegression(penalty = 'l2',C=1/regularization)
            model_lr.fit(X_train_inner,y_train_inner)
            
            y_test_est_i = model_lr.predict(X_test_inner)
            error_logistic_inner = np.sum(y_test_est_i != y_test_inner)/len(y_test_inner)
            error_lr_i.append(error_logistic_inner)
            
        # ANN
        for n_hidden_units in enumerate(n_hidden_units_l):
            print('\nHidden units:{}'.format(n_hidden_units))
            model_ann = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                torch.nn.Sigmoid()
                                )
            loss_fn = torch.nn.BCELoss()
            net, final_loss, learning_curve = train_neural_net(model_ann,
                                                               loss_fn,
                                                               X=X_train_inner,
                                                               y=y_train_inner,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            # Determine estimated class labels for test set
            y_sigmoid_inner = net(X_test_inner) # activation of final note, i.e. prediction of network
            y_test_est_inner = (y_sigmoid_inner > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
            y_test_inner = y_test_inner.type(dtype=torch.uint8)
            # Determine errors and error rate
            e_inner = (y_test_est_inner != y_test_inner)
            error_ann = (sum(e_inner).type(torch.float)/len(y_test_inner)).data.numpy()
            error_ann_i.append(error_ann) # store error rate for current CV fold 
            
    

