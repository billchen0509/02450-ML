# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:38:52 2023

@author: L Bz
"""
#%% import packages
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import sklearn.linear_model as lm

import matplotlib.pyplot as plt
from prettytable import PrettyTable

import innerModelSelect
from toolbox_02450 import mcnemar

#%% load data
path = 'forest_fire.csv'
df = pd.read_csv(path, header = 1)
df = df.drop(index = [122,123,124])
df = df.drop(columns = ['day','month','year'])
df.columns.values[0] = 'Temp'
raw_data = df.values

#%% 
cols = range(0,9)
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

#%% Initialize 10*10 2-layer CV
K1 = 10
CV = model_selection.KFold(K1, shuffle=True,random_state=42)

# Values of lambda
lambdas = np.power(10.,range(-5,5))

Error_train_rlr = np.empty((K1,1))
Error_test_rlr = np.empty((K1,1))

Error_train_knn = np.empty((K1,1))
Error_test_knn = np.empty((K1,1))

Error_train_bl = np.empty((K1,1))
Error_test_bl = np.empty((K1,1))

weights_rlr = []
y_ests_rlr = []
y_ests_knn = []
y_ests_bl = []
y_true = []

w_rlr = np.empty((M,K1))
mu = np.empty((K1, M-1))
sigma = np.empty((K1, M-1))

lambda_l = []
knn_k = []

#%% outer CV
k=0

for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    y_true.append(y_test)
    
    # Select best lambda in Regularized Logistic Regression
    K2 = 10
    opt_lambda = innerModelSelect.rlr_validate_logistic(X_train, y_train, lambdas, K2)
    lambda_l.append(opt_lambda)
    
    # Select best no. of neighbors in KNN
    opt_k = innerModelSelect.KNNCV(X_train, y_train)
    knn_k.append(opt_k)
    
    # Standarization
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :]
    
    
    # logistic regression model
    # Fit logistic regression model with entire X_train and opt_lambda L2
    model = lm.LogisticRegression(penalty='l2', C=1.0 / opt_lambda, 
                                  max_iter = 1000, solver = 'newton-cholesky')
    model = model.fit(X_train,y_train)
    weights_rlr.append(model.coef_)
    # Evaluate training and test performance on best lambda
    rlr_pred_train = model.predict(X_train)
    rlr_pred_test = model.predict(X_test)
    Error_train_rlr[k] = 1 - accuracy_score(y_train, rlr_pred_train)
    Error_test_rlr[k] = 1 - accuracy_score(y_test, rlr_pred_test)
    y_ests_rlr.append(rlr_pred_test)


    # KNN
    # Fit KNN model with entire X_train and opt_k
    knclassifier = KNeighborsClassifier(n_neighbors=opt_k);
    knclassifier.fit(X_train, y_train);
    # Evaluate training and test performance on best k
    knn_pred_train = knclassifier.predict(X_train);
    knn_pred_test = knclassifier.predict(X_test);
    Error_train_knn[k] = 1 - accuracy_score(y_train, knn_pred_train)
    Error_test_knn[k] = 1 - accuracy_score(y_test, knn_pred_test)
    y_ests_knn.append(knn_pred_test)
    
    
    # Baseline
    # Evaluate training and test performance on baseline model
    most_common_class = np.argmax(np.bincount(y_train))
    bl_pred_train = np.full(len(X_train), most_common_class)
    bl_pred_test = np.full(len(X_test), most_common_class)
    Error_train_bl[k] = 1 - accuracy_score(y_train, bl_pred_train)
    Error_test_bl[k] = 1 - accuracy_score(y_test, bl_pred_test)
    y_ests_bl.append(bl_pred_test)    
    
    k+=1

#%% Visualization

plt.figure(figsize=(12,8))
plt.plot(range(1, K1+1), Error_train_rlr, '-b')
plt.plot(range(1, K1+1), Error_test_rlr, '-r')
plt.plot(range(1, K1+1), Error_train_knn, ':b')
plt.plot(range(1, K1+1), Error_test_knn, ':r')
plt.plot(range(1, K1+1), Error_train_bl, '--b')
plt.plot(range(1, K1+1), Error_test_bl, '--r')

plt.xlabel('K Fold')
plt.ylabel('Generalization Error')
plt.legend(['Train Error - Regularized Logistic Regression',
            'Test Error - Regularized Logistic Regression', 
            'Train Error - KNN', 
            'Test Error - KNN', 
            'Train Error - Baseline', 
            'Test Error - Baseline'])
plt.grid()
plt.show()

# Directly Output
print('Regularized logistic regression:')
print('- Train error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('KNN:')
print('- Train error: {0}'.format(Error_train_knn.mean()))
print('- Test error:     {0}'.format(Error_test_knn.mean()))
ave_w = np.mean(np.concatenate(weights_rlr, axis=0), axis=0)
formatted_weights = ['{:.3f}'.format(weight) for weight in ave_w]
print('Averaged Weights of RLR:     {0}'.format(formatted_weights))

# Print Table
df1 = pd.DataFrame({"Outer fold": list(range(1, 11)),
                    "KNN k": knn_k, 
                    "KNN E^test_i": Error_test_knn.flatten(), 
                    "Logistic regression lambda": lambda_l, 
                    "Logistic regression E^test_i": Error_test_rlr.flatten(), 
                    "Baseline E^test_i": Error_test_bl.flatten() })
df1["Outer fold"] = df1["Outer fold"] + 1

print(tabulate(df1, headers='keys', tablefmt='github'))

#%% Statistical Evaluation 
y_est_rlr = np.array([item for sublist in y_ests_rlr for item in sublist])
y_est_knn = np.array([item for sublist in y_ests_knn for item in sublist])
y_est_bl = np.array([item for sublist in y_ests_bl for item in sublist])
y_true = np.array([item for sublist in y_true for item in sublist])

alpha = 0.05
# Group 1, rlr VS. knn
[thetahat1, CI1, p1] = mcnemar(y_true, y_est_rlr, y_est_knn, alpha=alpha)

# Group 2, rlr VS. bl
[thetahat2, CI2, p2] = mcnemar(y_true, y_est_rlr, y_est_bl, alpha=alpha)

# Group 3, knn VS. bl
[thetahat3, CI3, p3] = mcnemar(y_true, y_est_knn, y_est_bl, alpha=alpha)


CI_list = [CI1,CI2,CI3]
CI_format = [tuple("{:.4e}".format(x) for x in CI) for CI in CI_list]
p_list = [p1,p2,p3]
p_format = ["{:.4e}".format(p) for p in p_list]

table1 = PrettyTable()
table1.field_names = ['CI_rlr_knn','CI_rlr_bl','CI_knn_bl']
table1.add_row(CI_format)
print(table1)

table2 = PrettyTable()
table2.field_names = ['p_rlr_knn','p_rlr_bl','p_knn_bl']
table2.add_row(p_format)
print(table2)














