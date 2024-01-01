#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 23:29:37 2023

@author: billhikari
"""
# import packages
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid,plot,hist)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from toolbox_02450 import rlr_validate,train_neural_net, draw_neural_net,visualize_decision_boundary
from scipy import stats
import torch
from prettytable import PrettyTable
from scipy.stats import pearsonr
# load data
path = '/Users/Jialu/Downloads/Algerian_forest_fires_dataset_UPDATE.csv'

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

y_fwi = X[:,-1]
X_fwi = X[:,:9]

#%%
import matplotlib.pyplot as plt
# implement two-level cross-validation
y = y_fwi.reshape(-1,1)

# Normalize data
X = stats.zscore(X_fwi);
N,M = X.shape

#----------cross validation----------
K = 10        
CV = model_selection.KFold(K, shuffle=True,random_state=42)

# ----------linear regression----------
X_lr = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M_lr = M+1

# Values of lambda
lambdas = np.power(10.,range(-5,9))

Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
# linear regression without features
#Error_train_nofeatures = np.empty((K, 1))
#Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M_lr,K))
mu = np.empty((K, M))
sigma = np.empty((K, M))

lambda_l = []

# ----------ANN----------
# Parameters for neural network 
#n_hidden_units = 9     # number of hidden units
n_replicates = 1       # number of networks trained in each k-fold
max_iter = 10000 

opt_hidden_units = []
n_hidden_units_l = [1,2,4,8,16,32,64,128,256]


# ----------baseline----------

baseline_errors_test = np.empty(K)
#%%
# Outer crossvalidation loop
for n, n_hidden_units in enumerate(n_hidden_units_l):
    print('\nHidden units:{}'.format(n_hidden_units))
    
    model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
    pccs = []
    p_values = []
    errors = []
    baseline = []
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
        # linear regression
        X_train_lr = X_lr[train_index]
        y_train_lr = y[train_index]
        X_test_lr = X_lr[test_index]
        y_test_lr = y[test_index]
        internal_cross_validation = 10    
        
        mu[k, :] = np.mean(X_train_lr[:, 1:], axis=0)
        sigma[k, :] = np.std(X_train_lr[:, 1:], axis=0)
        
        X_train_lr[:, 1:] = (X_train_lr[:, 1:] - mu[k, :]) / sigma[k, :]
        X_test_lr[:, 1:] = (X_test_lr[:, 1:] - mu[k, :]) / sigma[k, :]
        
        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train_lr, y_train_lr, lambdas, internal_cross_validation)
        
        Xty = X_train_lr.T @ y_train_lr
        XtX = X_train_lr.T @ X_train_lr
        
        lambda_l.append(opt_lambda)
        lambdaI = opt_lambda * np.eye(M_lr)
        lambdaI[0,0] = 0
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        
        Error_train_rlr[k] = np.sum(np.square(y_train_lr - (X_train_lr @ w_rlr[:, k]).reshape((len(y_train_lr), 1))), axis=0) / np.shape(y_train_lr)[0]
        Error_test_rlr[k] = np.sum(np.square(y_test_lr - (X_test_lr @ w_rlr[:, k]).reshape((len(y_test_lr), 1))), axis=0) / np.shape(y_test_lr)[0]
        
        # baseline
        y_train_mean = np.mean(y[train_index])
        baseline_test_predict = np.full((len(test_index),1),y_train_mean)
        baseline_errors_test[k] = np.mean(np.square(y[test_index] - baseline_test_predict))
        # ANN
        # outer cross validation
        print('\nOuter crossvalidation fold: {0}/{1}'.format(k+1,K))
        # Extract training and testing set for current CV fold, convert to tensors
        X_train_outer = torch.Tensor(X[train_index,:])
        y_train_outer = torch.Tensor(y[train_index])
        X_test_outer = torch.Tensor(X[test_index,:])
        y_test_outer = torch.Tensor(y[test_index])
          
        error_i = None
        
        # Inner loop for hyperparameter tuning
        for (j, (train_index_inner, test_index_inner)) in enumerate(CV.split(X_train_outer,y_train_outer)):
            print('\nInner crossvalidation fold: {0}/{1}'.format(j+1,K))
              
            # Extract training and testing set for current CV fold, convert to tensors
            X_train_inner = X_train_outer[train_index_inner,:]
            y_train_inner = y_train_outer[train_index_inner]
            X_test_inner = X_train_outer[test_index_inner,:]
            y_test_inner = y_train_outer[test_index_inner]
            
            loss_fn = torch.nn.MSELoss()
            
                # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_inner,
                                                               y=y_train_inner,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            print('\n\tBest loss: {}\n'.format(final_loss))
                # Determine estimated class labels for test set
            y_test_est_inner = net(X_test_inner)
            
                # Determine errors and errors
            se_inner = (y_test_est_inner.float()-y_test_inner.float())**2 # squared error
            mse_inner = (sum(se_inner).type(torch.float)/len(y_test_inner)).data.numpy() #mean
            
            pcc_inner, p_value_inner = pearsonr(y_test_est_inner.detach().numpy().T.tolist()[0], y_test_inner.detach().numpy().T.tolist()[0])
            
            if error_i == None:
                error_i = (mse_inner, [n_hidden_units, k, j])
            elif mse_inner < error_i[0]:
                error_i = (mse_inner, [n_hidden_units, k, j])
            
            torch.save(net, f'model_{n_hidden_units}_{k}_{j}.pth')
            with open('model_performance.txt', 'a') as f:
                f.write('mse_model' + '\t' + f'{n_hidden_units},{k},{j}' + '\t' + str(mse_inner) + '\t' + str(pcc_inner) + '\t' + str(p_value_inner) + '\n')
        
        best_model_fold = error_i[1]
        best_model_index = ['b_n','b_k','b_j']
        
        for i in range(len(best_model_fold)):
            exec(f'{best_model_index[i]} = {best_model_fold[i]}')
        
        best_model = torch.load(f'model_{b_n}_{b_k}_{b_j}.pth')
        best_model.eval()
        
        X_test_outer_tensor = X_test_outer
        
        y_test_outer_tensor = y_test_outer
        
        with torch.no_grad():
            y_test_est_outer = best_model(X_test_outer_tensor)
        
        # Determine errors and errors
        se_outer = (y_test_est_outer-y_test_outer_tensor)**2 # squared error
        mse_outer = torch.mean(se_outer) #mean
        errors.append(mse_outer)
        # baseline
        baseline.append(np.mean(baseline_errors_test))
        
        
        pcc_outer, p_value_outer = pearsonr(y_test_est_outer.detach().numpy().T.tolist()[0], y_test_est_outer.detach().numpy().T.tolist()[0])
        pccs.append(pcc_outer)
        p_values.append(p_value_outer)
    
    # print table
    table = PrettyTable()
    table.add_column('Outer fold', [i for i in range(1, K+1)])
    table.add_column('h*', [n_hidden_units for i in range(1, K+1)])
    table.add_column('ann_E', errors)
    table.add_column('lambda*', lambda_l)
    table.add_column('lr_E', Error_test_rlr.squeeze().tolist())
    table.add_column('baseline', baseline)
    table.add_column('PCC', pccs)
    table.add_column('p-value', p_values)
    with open('performance_table.txt', 'a') as f:
        f.write(str(table))
        f.write('\n' + '='*40 + '\n')

