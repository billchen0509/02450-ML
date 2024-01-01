# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:44:33 2019

@author: Lychen
"""
import numpy as np
from sklearn import linear_model, model_selection
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
import matplotlib.pyplot as plt
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


'''
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
'''
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

#attributeNames_n = attributeNames[4:10:4]   # [4:10:4] & [5:10:3]


X = data_r[:,1:].astype(np.float)         # data matix
X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
N = len(data_r[:,0])       # number of observations
M = len(attributeNames)  # number of attributes
C = len(classLabels)     # number of class

K=2
CV = model_selection.KFold(K, shuffle=True)

# ----------logr---------
reg_list = [pow(10,i) for i in range(-5,6)]
opt_reg = []
error_lr = []

# ----------ANN----------
n_hidden_units_l = [i for i in range(1, 11)]
opt_hidden = []
error_ann = []
n_replicates = 3
max_iter = 1000

# ----------basline--------
error_bl = []



for (k, (train_index, test_index)) in enumerate(CV.split(X, y)):
    
    # ----------logistic regression----------
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    n_hidden_units = 5 # number of hidden units in the signle hidden layer
    model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                            torch.nn.ReLU(), # 1st transfer function
                            # Output layer:
                            # H hidden units to C classes
                            # the nodes and their activation before the transfer 
                            # function is often referred to as logits/logit output
                            torch.nn.Linear(n_hidden_units, C), # C logits
                            # To obtain normalised "probabilities" of each class
                            # we use the softmax-funtion along the "class" dimension
                            # (i.e. not the dimension describing observations)
                            torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit output
                            )
    # Since we're training a multiclass problem, we cannot use binary cross entropy,
    # but instead use the general cross entropy loss:
    loss_fn = torch.nn.CrossEntropyLoss()
    # Train the network:
    net, _, _ = train_neural_net(model, loss_fn,
                             X=torch.tensor(X_train, dtype=torch.float),
                             y=torch.tensor(y_train, dtype=torch.long),
                             n_replicates=3,
                             max_iter=10000)
    # Determine probability of each class using trained network
    softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
    # Get the estimated class as the class with highest probability (argmax on softmax_logits)
    y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy() 
    # Determine errors
    e = (y_test_est != y_test)
    print('Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(sum(e),len(e)))

    predict = lambda x:  (torch.max(net(torch.tensor(x, dtype=torch.float)), dim=1)[1]).data.numpy() 
    plt.figure(1,figsize=(9,9))
    visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
    plt.title('ANN decision boundaries')

    plt.show()
    
    '''
    error_lr_i = np.empty((K, len(reg_list)))
    error_ann_i = np.empty((K, len(n_hidden_units_l)))
    
    for (k_i, (train_index_i, test_index_i)) in enumerate(CV.split(X_train, y_train)):
        X_train_i = X_train[train_index_i,:]
        y_train_i = y_train[train_index_i]
        X_test_i = X_train[test_index_i,:]
        y_test_i = y_train[test_index_i]
        
        #X_train_ann_i = torch.tensor(X_train_i)#, dtype=torch.float)
        #y_train_ann_i = torch.tensor(X)
        
        
        for i, regularization_strength in enumerate(reg_list):
            model_lr = linear_model.LogisticRegression(
                    solver='lbfgs', multi_class='multinomial', 
                    tol=1e-4, random_state=1, 
                    penalty='l2', C=1/regularization_strength)    
            y_test_i_est = model_lr.fit(X_train_i, y_train_i).predict(X_test_i)
            error_lr_i[k, i] = np.sum(y_test_i_est!=y_test_i) / len(y_test_i)
        
        
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
    regularization_strength = reg_list[np.unravel_index(np.argmin(error_lr_i), error_lr_i.shape)[1] + 1]
    model_lr = linear_model.LogisticRegression(
            solver='lbfgs', multi_class='multinomial', 
            tol=1e-4, random_state=1, 
            penalty='l2', C=1/regularization_strength)
    y_test_lr_est = model_lr.fit(X_train, y_train).predict(X_test)
    opt_reg.append(regularization_strength)
    error_lr.append(np.sum(y_test_lr_est != y_test) / len(y_test))
    
    # ann
    n_hidden_units = n_hidden_units_l[np.unravel_index(np.argmin(error_ann_i), error_ann_i.shape)[1] + 1]
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
    
    # bl
    error_bl.append(stats.mode(y_test)[1][0]/len(y_test))
    
    
    k = k + 1
    '''

'''
table = PrettyTable()
table.add_column('K', [i for i in range(1,K+1)])
table.add_column('h*', opt_hidden)
table.add_column('E_ann', error_ann)
table.add_column('lambda*', opt_reg)
table.add_column('E_lr', error_lr)
table.add_column('baseline', error_bl)
print(table)
'''
    
    
    
    
    
    
    
    
    
    
    
