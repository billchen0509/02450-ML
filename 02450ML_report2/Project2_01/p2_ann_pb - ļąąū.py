# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:44:33 2019

@author: Lychen
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
import torch

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
y = np.asarray(classLabels[:]).reshape(len(classLabels),1)

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

X = stats.zscore(X)

# -----PCA-----
do_pca_preprocessing = False
if do_pca_preprocessing:
    Y = stats.zscore(X,0);
    U,S,V = np.linalg.svd(Y,full_matrices=False)
    V = V.T
    #Components to be included as features
    k_pca = 3
    X = X @ V[:,0:k_pca]
    N, M = X.shape

n_hidden_units = 1
n_replicates = 1
max_iter = 5000

K = 10
CV = model_selection.KFold(K, shuffle=True)

summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))

color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

model = lambda: torch.nn.Sequential(torch.nn.Linear(M, n_hidden_units), torch.nn.Tanh(), torch.nn.Linear(n_hidden_units, 1),)

loss_fn = torch.nn.MSELoss()

print('Training model of type:\n\n{}\n.'.format(str(model())))
errors = []

for(k, (train_index, test_index)) in enumerate(CV.split(X, y)):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1, K))
    
    X_train = torch.tensor(X[train_index,:], dtype=torch.float)
    y_train = torch.tensor(y[train_index], dtype=torch.float)
    X_test = torch.tensor(X[test_index,:], dtype=torch.float)
    y_test = torch.tensor(y[test_index], dtype=torch.uint8)
    
    net, final_loss, learning_curve = train_neural_net(model, loss_fn, X=X_train, y=y_train, n_replicates=n_replicates, max_iter=max_iter)
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    y_test_est = net(X_test)
    
    se = (y_test_est.float()-y_test.float())**2
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy()
    errors.append(mse)
    
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold')
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('MSE')
summaries_axes[1].set_title('Test mean-squared-error')

print('Diagram of best neural net is last old:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf = [str(net[i]) for i in [1,2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))
plt.figure(figsize=(10,10))
y_est = y_test_est.data.numpy()
y_true = y_test.data.numpy()
axis_range = [np.min([y_est, y_true])-1, np.max([y_est, y_true])+1]
plt.plot(axis_range, axis_range, 'k--')
plt.plot(y_true, y_est, 'ob', alpha=0.25)
plt.legend(['Perfect estimation','Model estimation'])
plt.title('Abalone: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range)
plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()
plt.show()
