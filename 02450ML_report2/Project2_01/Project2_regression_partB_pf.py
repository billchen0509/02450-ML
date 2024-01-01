# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:44:33 2019

@author: Lychen
"""
import numpy as np
from sklearn import model_selection
from toolbox_02450 import rlr_validate, train_neural_net
from scipy import stats
import torch
from prettytable import PrettyTable

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

data[:,3:-1] = data_r[:,1:-1]
data[:,-1] = data_r[:,-1]

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



# ----------cross validation----------
K = 3
CV = model_selection.KFold(K, shuffle=True)

# ----------linear regression----------
X_lr = np.concatenate((np.ones((N,1)), X), 1)
attributeNames = [u'Offset'] + attributeNames
M_lr = M + 1

lambdas = np.power(10., range(-5, 9))

#Error_train = np.empty((K, 1))
#Error_test = np.empty((K, 1))
Error_train_reg = np.empty((K, 1))
Error_test_reg = np.empty((K, 1))
#Error_train_nofeatures = np.empty((K, 1))
#Error_test_nofeatures = np.empty((K, 1))
#Error_train_fs = np.empty((K, 1))
#Error_test_fs = np.empty((K, 1))
#Features = np.zeros((M, K))
mu = np.empty((K, M))
sigma = np.empty((K, M))
#w = np.empty((M_lr, K))
w_reg = np.empty((M_lr, K))
lambdas_l = []
z_lr = []

# ----------ANN parameters----------
#n_hidden_units = 1
n_replicates = 1
max_iter = 1000

errors = []
opt_hidden_units = []
final_loss_l = []
n_hidden_units_l = [1,2,3]

z_ann = []

X_ann = stats.zscore(X)

# ----------baseline----------
baseline = []
z_bl = []


#summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))

#color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

#model = lambda: torch.nn.Sequential(torch.nn.Linear(M, n_hidden_units), torch.nn.Tanh(), torch.nn.Linear(n_hidden_units, 1),)

#loss_fn = torch.nn.MSELoss()

#print('Training model of type:\n\n{}\n.'.format(str(model())))
k=0
for(k, (train_index, test_index)) in enumerate(CV.split(X, y)):
    
    # ----------Linear regression---------
    X_train_lr = X_lr[train_index]
    X_test_lr = X_lr[test_index]
    y_train_lr = y[train_index]
    y_test_lr = y[test_index]
    internal_cross_validation = 10
    
    #mu[k, :] = np.mean(X_train_lr[:, 1:], axis=0)
    #sigma[k, :] = np.std(X_train_lr[:, 1:], axis=0)
    
    #X_train_lr[:, 1:] = (X_train_lr[:, 1:] - mu[k, :]) / sigma[k, :]
    #X_test_lr[:, 1:] = (X_test_lr[:, 1:] - mu[k, :]) / sigma[k, :]
    
    #Error_train_nofeatures[k] = np.sum(np.square(y_train_lr - np.mean(y_train_lr)), axis=0) / np.shape(y_train_lr)[0]
    #Error_test_nofeatures[k] = np.sum(np.square(y_test_lr - np.mean(y_train_lr)), axis=0) / np.shape(y_test_lr)[0]
    #w[:, k] = np.squeeze(np.linalg.solve(X_train_lr.T @ X_train_lr, X_train_lr.T @ y_train_lr))
    
    #Error_train[k] = np.sum(np.square(y_train_lr - X_train_lr @ w[:,k]), axis=0) / np.shape(y_train_lr)[0]
    #Error_test[k] = np.sum(np.square(y_test_lr - X_test_lr @ w[:, k]), axis=0) / np.shape(y_test_lr)[0]
    
    # -----linear regression with regularization-----
    opt_var_err, opt_lambda, mean_w_lambda, train_err_lambda, test_err_lambda = rlr_validate(X_train_lr, y_train_lr, lambdas, internal_cross_validation)

    lambdas_l.append(opt_lambda)
    lambdaI = opt_lambda * np.eye(M_lr)
    lambdaI[0,0] = 0
    
    w_reg[:, k] = np.squeeze(np.linalg.solve((X_train_lr.T @ X_train_lr + lambdaI), (X_train_lr.T @ y_train_lr)))
    
    z_lr.append(np.abs(y_test_lr - (X_test_lr @ w_reg[:, k]).reshape((len(y_test_lr), 1)))**2)
    
    Error_train_reg[k] = np.sum(np.square(y_train_lr - (X_train_lr @ w_reg[:, k]).reshape((len(y_train_lr), 1))), axis=0) / np.shape(y_train_lr)[0]
    Error_test_reg[k] = np.sum(np.square(y_test_lr - (X_test_lr @ w_reg[:, k]).reshape((len(y_test_lr), 1))), axis=0) / np.shape(y_test_lr)[0]
    
    
    # ----------ANN----------
    print('\nOuter Crossvalidation fold: {0}/{1}'.format(k+1, K))
    
    X_train_ann = torch.tensor(X_ann[train_index,:], dtype=torch.float)
    y_train_ann = torch.tensor(y[train_index], dtype=torch.float)
    X_test_ann = torch.tensor(X_ann[test_index,:], dtype=torch.float)
    y_test_ann = torch.tensor(y[test_index], dtype=torch.uint8)
    
    error_i = np.empty((K, len(n_hidden_units_l)))
    
    # -----inner cross validation for ann-----
    for (j, (train_index_i, test_index_i)) in enumerate(CV.split(X_train_ann, y_train_ann)):
        print('\n\tInner Crossvalidation fold: {0}/{1}'.format(j+1, K))
        X_train_i = torch.tensor(X_train_ann[train_index_i,:], dtype=torch.float)
        y_train_i = torch.tensor(y_train_ann[train_index_i], dtype=torch.float)
        X_test_i = torch.tensor(X_train_ann[test_index_i,:], dtype=torch.float)
        y_test_i = torch.tensor(y_train_ann[test_index_i], dtype=torch.uint8)
        
        #error_nhu = np.empty((K, 5))
        loss_fn = torch.nn.MSELoss()
        for n_hidden_units in n_hidden_units_l:
            print('\n\tHidden units: {}'.format(n_hidden_units))
            model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), 
                    torch.nn.Tanh(), 
                    torch.nn.Linear(n_hidden_units, 1))
            #print('\n\tTraining model of type:\n\n{}\n.'.format(str(model())))
            net, final_loss, learning_curve = train_neural_net(
                    model, loss_fn, X=X_train_i, y=y_train_i, 
                    n_replicates=n_replicates, max_iter=max_iter)
            print('\n\tBest loss: {}\n'.format(final_loss))
            y_test_est_i = net(X_test_i)
        
            se = (y_test_est_i.float()-y_test_i.float())**2
            mse = (sum(se).type(torch.float)/len(y_test_i)).data.numpy()
            error_i[j, n_hidden_units-1] = mse
            
    n_hidden_units = np.unravel_index(np.argmin(error_i), np.asarray(error_i).shape)[1] + 1
    opt_hidden_units.append(n_hidden_units)
    
    model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, n_hidden_units), 
            torch.nn.Tanh(), 
            torch.nn.Linear(n_hidden_units, 1))
    
    final_loss_l.append(final_loss)
    
    print('Optimal hidden units: {} for outer CV: {}'.format(n_hidden_units, k+1))
    net, final_loss, learning_curve = train_neural_net(model, loss_fn, X=X_train_ann, y=y_train_ann, n_replicates=n_replicates, max_iter=max_iter)
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    y_test_est = net(X_test_ann)
    
    se = (y_test_est.float()-y_test_ann.float())**2
    mse = (sum(se).type(torch.float)/len(y_test_i)).data.numpy()
    errors.append(mse[0])
    z_ann.append(se.detach().numpy())
    
    
    # ----------baseline----------
    baseline.append(np.sum(y[test_index])/len(y[test_index]))
    
    z_bl.append(y[test_index])#**2
    
    '''
    # ----------performance----------

    # CI of performance difference, H0: A and B have the same performance
    CI_data.append([])
    p_data.append([])
    z_list = [z_lr, z_ann.detach().numpy(), z_bl]
    for i, j in enumerate([1,2,0]):
        z = z_list[i] - z_list[j]
        CI = stats.t.interval(1-alpha, df=len(z)-1, loc=np.mean(z), scale=stats.sem(z))
        #CI_data[k, i] = CI
        CI_data[k].append([round(CI[l][0],4) for l in range(len(CI))])
        p = stats.t.cdf(-np.mean(z)/stats.sem(z), df=len(z)-1)
        #p_data[k, i] = p
        p_data[k].append(p)
        '''
    
    
    k = k + 1

# print table
table = PrettyTable()
table.add_column('Outer fold', [i for i in range(1, K+1)])
table.add_column('h*', opt_hidden_units)
table.add_column('ann_E', errors)
table.add_column('lambda*', lambdas_l)
table.add_column('lr_E', [round(i, 4) for i in Error_test_reg.squeeze().tolist()])
table.add_column('baseline', baseline)
'''
table.add_column('CI_lr_ann', [CI_data[i][0] for i in range(K)])
table.add_column('CI_lr_bl', [CI_data[i][1] for i in range(K)])
table.add_column('CI_ann_bl', [CI_data[i][2] for i in range(K)])
table.add_column('p_lr_ann', [p_data[i][0] for i in range(K)])
table.add_column('p_lr_bl', [p_data[i][1] for i in range(K)])
table.add_column('p_ann_bl', [p_data[i][2] for i in range(K)])
'''
print(table)



# ----------performance----------
alpha = 0.05
CI_data = []
p_data = []

#z_ann = [z_ann[i].detach().numpy() for i in range(len(z_ann))]
z_list = [z_lr, z_ann, z_bl]
for i, j in enumerate([1,2,0]):
    z =(np.sum(np.asarray(z_list[i]), axis=0) - np.sum(np.asarray(z_list[j]), axis=0))#/len(test_index)
    CI = stats.t.interval(1-alpha, df=len(z)-1, loc=np.mean(z), scale=stats.sem(z))
    #CI_data[k, i] = CI
    CI_data.append([round(CI[l][0],4) for l in range(len(CI))])
    p = stats.t.cdf(-np.mean(z)/stats.sem(z), df=len(z)-1)
    #p_data[k, i] = p
    p_data.append(p)


table2 = PrettyTable()
table2.field_names = ['CI_lr_ann','CI_lr_bl','CI_ann_bl']
table2.add_row(CI_data)
print(table2)

table3 = PrettyTable()
table3.field_names = ['p_lr_ann','p_lr_bl','p_ann_bl']
table3.add_row([i.squeeze().tolist() for i in p_data])
print(table3)



alpha = 0.05
CI_data_2 = []
p_data_2 = []

l = lambdas_l[Error_test_reg.tolist().index(min(Error_test_reg))]
h = opt_hidden_units[errors.index(min(errors))]
X_train_p, X_test_p, y_train_p, y_test_p = model_selection.train_test_split(X,y,test_size=0.33, random_state=42)

lambdaI = l * np.eye(M_lr)
X_train_p_lr = np.concatenate((np.ones((len(X_train_p),1)), X_train_p), 1)
X_test_p_lr = np.concatenate((np.ones((len(X_test_p),1)), X_test_p), 1)
w = np.squeeze(np.linalg.solve((X_train_p_lr.T @ X_train_p_lr + lambdaI), (X_train_p_lr.T @ y_train_p)))
y_est_lr = np.abs(y_test_p - (X_test_p_lr @ w).reshape((len(y_test_p), 1)))**2

model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, h), 
        torch.nn.Tanh(), 
        torch.nn.Linear(h, 1))
loss_fn = torch.nn.MSELoss()
net, final_loss, learning_curve = train_neural_net(
        model, loss_fn, 
        X=torch.tensor(X_train_p, dtype=torch.float),
        y=torch.tensor(y_train_p, dtype=torch.float),
        n_replicates=n_replicates, max_iter=max_iter)
y_est_ann = ((net(torch.tensor(X_test_p, dtype=torch.float)).float()).detach().numpy()-y_test_p)**2

y_est_bl = y_test_p

#z_ann = [z_ann[i].detach().numpy() for i in range(len(z_ann))]
y_est_list = [y_est_lr, y_est_ann, y_est_bl]
for i, j in enumerate([1,2,0]):
    z =y_est_list[i] - y_est_list[j]#/len(test_index)
    CI = stats.t.interval(1-alpha, df=len(z)-1, loc=np.mean(z), scale=stats.sem(z))
    #CI_data[k, i] = CI
    CI_data_2.append(CI)
    p = stats.t.cdf(-np.mean(z)/stats.sem(z), df=len(z)-1)
    #p_data[k, i] = p
    p_data_2.append(p)

table4 = PrettyTable()
table4.field_names = ['CI_lr_ann','CI_lr_bl','CI_ann_bl']
table4.add_row(CI_data_2)
print(table4)

table5 = PrettyTable()
table5.field_names = ['p_lr_ann','p_lr_bl','p_ann_bl']
table5.add_row([i.squeeze().tolist() for i in p_data_2])
print(table5)

