#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:48:27 2023

@author: billhikari
"""

#%%
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

#%%
# load data
path = '/Users/billhikari/Documents/02450 ML/02450ML_report1/forest_fire.csv'

df = pd.read_csv(path, header = 1)

df = df.drop(index = [122,123,124])

df = df.drop(columns = ['day','month','year'])

df.columns.values[0] = 'Temp'

raw_data = df.values

#%% 
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

# Data standardization
#Y = X - np.ones((N,1))*X.mean(axis=0)
#Y = Y / np.std(Y,0)

#%% 
# project 2: Regression part a
# a.1
# Extract values from data since we don't consider 'fire/not fire' right now
y_fwi = X[:,-1]
X_fwi = X[:,:9]

# update N,M 
N,M = X_fwi.shape

residual = []

# mse = 1.47 we consider this result is good
# The scatter plot seems to show a positive linear relationship between the true and predicted values, 
# which indicates that our model has captured some underlying patterns in the data.
# Most of the points cluster around a diagonal line, suggesting a decent model fit for many observations.
# The residuals are mostly centered around 0, indicating that, on average, the model neither consistently overpredicts nor underpredicts the FWI.
# The distribution of residuals appears to be approximately normal, which is a good sign for our linear regression model.
#%%
# project 2: Regression part a
# a.2
X_fwi = stats.zscore(X_fwi)
# Add offset attribute
X_fwi = np.concatenate((np.ones((X_fwi.shape[0],1)),X_fwi),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

# Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True,random_state=42)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))

lambda_l = []

#%%
k=0
for train_index, test_index in CV.split(X_fwi,y_fwi):
    
    # extract training and test set for current CV fold
    X_train = X_fwi[train_index]
    y_train = y_fwi[train_index]
    X_test = X_fwi[test_index]
    y_test = y_fwi[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Estimate weights for the optimal value of lambda, on entire training set
    lambda_l.append(opt_lambda)
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    
    #model = linear.fit(X_train, y_train)
    #Error_train_rlr[k] = np.square(y_train-model.predict(X_train)).sum()/y_train.shape[0]
    #Error_test_rlr[k] = np.square(y_test-model.predict(X_test)).sum()/y_test.shape[0]
    
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_est = model.predict(X_train)
    residual.append((y_est - y_train).tolist())

    k+=1

show()
# Display results
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))

print('Weights in last fold:')
for m in range(M-1):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
# We will choose our regularization parameter lambda = 1,
# where the validation error is at its lowest. 
# Beyond this point, as the regularization factor increases further, 
# the validation error starts to rise significantly.


figure(k, figsize=(12,8))
subplot(1,2,1)
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
    # You can choose to display the legend, but it's omitted for a cleaner 
    # plot, since there are many attributes
    #legend(attributeNames[:], loc='best')
opt_l = lambda_l[np.argmin(test_err_vs_lambda)]
subplot(1,2,2)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_l)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()



# Display scatter plot
figure()
subplot(2,1,1)
plot(y_est, y_train, '.')
xlabel('FWI (true)'); ylabel('FWI (estimated)');
subplot(2,1,2)
hist(residual,100)

show()
