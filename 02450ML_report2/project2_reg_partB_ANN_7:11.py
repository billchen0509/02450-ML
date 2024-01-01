#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:35:07 2023

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

y_fwi = X[:,-1]
X_fwi = X[:,:9]

#%%
import matplotlib.pyplot as plt
# implement two-level cross-validation
X_fwi = X_fwi[:,1:]
N, M = X_fwi.shape
y = y_fwi.reshape(-1,1)

# Normalize data
X = stats.zscore(X_fwi);

#----------cross validation----------
K = 5          
CV = model_selection.KFold(K, shuffle=True,random_state=42)

# ----------linear regression----------
X_lr = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames


# Values of lambda
lambdas = np.power(10.,range(-5,9))

Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
# linear regression without features
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))

lambda_l = []

# ----------ANN----------
# Parameters for neural network 
n_hidden_units = 1     # number of hidden units
n_replicates = 1       # number of networks trained in each k-fold
max_iter = 10000 

opt_hidden_units = []
final_loss_l = []
#n_hidden_units_l = [1,2,3,4,5,6,7,8,9,10]

errors = []

# ----------baseline----------
#%%
# Make figure for holding summaries (errors and learning curves)
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
# Define the model
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )

loss_fn = torch.nn.MSELoss()

# make a list for storing generalizaition error in each loop

# Outer crossvalidation loop
for k, (train_index, test_index) in enumerate(CV.split(X,y)):
    print('\nOuter crossvalidation fold: {0}/{1}'.format(k+1,K))
    
    # Extract training and testing set for current CV fold, convert to tensors
    X_train_outer = torch.Tensor(X[train_index,:])
    y_train_outer = torch.Tensor(y[train_index])
    X_test_outer = torch.Tensor(X[test_index,:])
    y_test_outer = torch.Tensor(y[test_index])
      
    best_inner_error = np.inf
    best_model = None
    
    # Inner loop for hyperparameter tuning
    for j, (train_index_inner, test_index_inner) in enumerate(CV.split(X_train_outer.numpy(),y_train_outer.numpy())):
        print('\nInner crossvalidation fold: {0}/{1}'.format(j+1,K))
          
        # Extract training and testing set for current CV fold, convert to tensors
        X_train_inner = X_train_outer[train_index_inner]
        y_train_inner = y_train_outer[train_index_inner]
        X_test_inner = X_train_outer[test_index_inner]
        y_test_inner = y_train_outer[test_index_inner]
        
        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train_inner,
                                                           y=y_train_inner,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        #print('\n\tBest loss: {}\n'.format(final_loss))
        # Determine estimated class labels for test set
        y_test_est_inner = net(X_test_inner)
        
        # Determine errors and errors
        se_inner = (y_test_est_inner.float()-y_test_inner.float())**2 # squared error
        mse_inner = (sum(se_inner).type(torch.float)/len(y_test_inner)).data.numpy() #mean
        #errors.append(mse_inner) # save the best model
    
    # After inner CV, train on entire outer train set with best model,test on outer test set
    # Determine estimated class labels for test set
    y_test_est_outer = net(X_test_outer)
    
    # Determine errors and errors
    se_outer = (y_test_est_outer.float()-y_test_outer.float())**2 # squared error
    mse_outer = (sum(se_outer).type(torch.float)/len(y_test_outer)).data.numpy() #mean
    errors.append(mse_outer)
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
# Display the MSE across outer folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Outer fold')
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('MSE')
summaries_axes[1].set_title('Generalization error across outer folds')
plt.show()


# Print the average error rate
print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))

# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of 
# the true/known value - these values should all be along a straight line "y=x", 
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value
plt.figure(figsize=(10,10))
y_est = y_test_est_outer.data.numpy(); y_true = y_test_outer.data.numpy()
axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('FWI: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()

plt.show()