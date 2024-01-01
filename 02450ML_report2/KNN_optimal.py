#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 15:16:07 2023

@author: billhikari
"""
#%%
# import packages
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import sklearn.linear_model as lm

from matplotlib.pyplot import (figure, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)
#%%
# load data
path = 'forest_fire.csv'
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
y = np.array([classDict[cl] for cl in classLabels])

# N: no. of samples; M: attributes
N,M = X.shape
C = len(classNames)
#%%
K1 = 10
CV = model_selection.KFold(K1, shuffle=True,random_state=42)

w_rlr = np.empty((M,K1))
mu = np.empty((K1, M-1))
sigma = np.empty((K1, M-1))

k=0

for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    # Select best lambda in Regularized Logistic Regression
    K2 = 10
   
    # Select best no. of neighbors in KNN
    opt_k = 3
    
    # Standarization
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :]
    
    
    
    
    # KNN
    # Fit KNN model with entire X_train and opt_k
    knclassifier = KNeighborsClassifier(n_neighbors=opt_k);
    knclassifier.fit(X_train, y_train);
    
    
    k+=1


#%%
y_est = knclassifier.predict(X_test)



# Plot the classfication results
styles = ['ob', 'or']
for c in range(C):
    class_mask = (y_est==c)
    plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
    plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
title('Synthetic data classification - KNN');


# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est);
accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
figure(2);
imshow(cm, cmap='binary', interpolation='None');
colorbar()
xticks(range(C)); yticks(range(C));
xlabel('Predicted class'); ylabel('Actual class');
title('Accuracy: {0}%, Error Rate: {1}%'.format(round(accuracy,4), round(error_rate,4)));

show()









