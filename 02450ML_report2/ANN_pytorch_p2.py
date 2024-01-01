#!/usr/bin/env python3
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.stats import pearsonr
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from torch.autograd import Variable
import time

from toolbox_02450 import rlr_validate,train_neural_net, draw_neural_net,visualize_decision_boundary

# %%
from pytorchtools import EarlyStopping

import os
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

path = '/Users/billhikari/Documents/02450 ML/02450ML_report1/forest_fire.csv'


def error(y, y_pred):
	return 0.5*(y_pred - y)**2


def load_file(path):
    data = pd.read_csv(path, header=1)

    data = data.drop(index=[122, 123, 124])

    data = data.drop(columns=['day', 'month', 'year'])

    data.columns.values[0] = 'Temp'

    # Define a function to remove whitespace from strings
    def remove_whitespace(s):
        return s.replace(" ", "")

    # Apply the function to all string columns using map
    data = data.apply(lambda col: col.map(remove_whitespace) if col.dtype == 'object' else col)

    raw_data = data.values

    return data, raw_data



def load_data(data, raw_data):
	cols = range(0,10)
	X = raw_data[:, cols]

	# convert X string array to float array
	X = np.asfarray(X, dtype = float)

	# extract the attribute names 
	attributeNames = np.asarray(data.columns[cols])

	# extract the last column
	classLabels = raw_data[:,-1]

	data.iloc[:,-1] = data.iloc[:,-1].apply(lambda x: x.strip())

	#determine the class labels
	classNames = np.flip(np.unique(classLabels))

	C = len(classNames)

	N,M = X.shape

	return X, C, N, M, attributeNames

def fwi_selection(X):
    y_fwi = X[:,-1]
    X_fwi = X[:,:9]
    return y_fwi, X_fwi



def mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def pcc(y_true, y_pred):
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    centered_true = y_true - mean_true
    centered_pred = y_pred - mean_pred
    pcc_num = torch.sum(centered_true * centered_pred)
    pcc_den = torch.sqrt(torch.sum(centered_true ** 2) * torch.sum(centered_pred ** 2))
    return pcc_num / pcc_den


def evaluate_model(x_eval, y_eval, net):
    net.eval()
    output = net(x_eval)
    return output

def random_split_outer(data, num_splits=10):
    splits = []

    for _ in range(num_splits):
        train, test = train_test_split(data, test_size=0.1, random_state=42)
        splits.append((train, test))

    return splits

def random_split_inner(data, num_splits=10):
    splits = []

    for _ in range(num_splits):
        train, test = train_test_split(data, test_size=0.1, random_state=42)
        splits.append((train, test))

    return splits

def Tensorprocess(x, y):
    x_valid = x.reshape(x.shape[0], -1)
    x_valid = Variable(torch.from_numpy(x_valid.astype('float32')))
    y_valid = Variable(torch.from_numpy(y.astype('float32'))).view(-1, 1)
    return x_valid, y_valid

# No mini-batch loading
# mini-batch loading




data, raw_data = load_file(path)
X, C, N, M, attributeNames = load_data(data, raw_data)
y_fwi, X_fwi = fwi_selection(X)




N_HIDDEN_NEURONS_param = [1,2,3,5,10,16,64]
n_replicates = 1       # number of networks trained in each k-fold
max_iter = 10000 
results_dir = "./results/"
K_1 = 10
K_2 = 10


if not os.path.exists(results_dir):
	os.makedirs(results_dir)


split = random_split_outer(X)
for H in range(len(N_HIDDEN_NEURONS_param)):
	N_HIDDEN_NEURONS = N_HIDDEN_NEURONS_param[H]
	model = lambda: torch.nn.Sequential(torch.nn.Linear(M-1, N_HIDDEN_NEURONS),
	# M features to H hiden units
	torch.nn.Tanh(),        # 1st transfer function
	torch.nn.Linear(N_HIDDEN_NEURONS, 1)  # H hidden units to 1 output neuron# final tranfer function
	).to(device)
	errors = []
	summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
	color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
	'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
    
	for outer_index in range(K_1):
		print("\tOuter index: {}/10".format(outer_index + 1))
		evaluation_data = split[outer_index][1]
		y_fwi, X_fwi = fwi_selection(evaluation_data)
		x_evaluation, y_evaluation = Tensorprocess(X_fwi, y_fwi)
		x_evaluation, y_evaluation = x_evaluation.to(device), y_evaluation.to(device)
		inner_split = random_split_inner(split[outer_index][0])

		for inner_index in range(K_2):
			validation_data = inner_split[inner_index][1]
			y_fwi, X_fwi = fwi_selection(validation_data)
			x_validation, y_validation = Tensorprocess(X_fwi, y_fwi)
			x_validation, y_validation = x_validation.to(device), y_validation.to(device)

			train_data = inner_split[inner_index][0]
			y_fwi, X_fwi = fwi_selection(train_data)
			x_train, y_train = Tensorprocess(X_fwi, y_fwi)
			x_train, y_train = x_train.to(device), y_train.to(device)
			#batch_size = x_train.shape[0]
			#n_features = x_train.shape[1]
			print("\t\t\tTraining ANN ...Hidden neuron param: {}/7, Outer index: {}/10, Inner index: {}/10".format(H+1, outer_index+1, inner_index+1))
			#net = Net(n_features, N_HIDDEN_NEURONS)
			#optimizer = optim.SGD(net.parameters(), lr=learning_rate)
			loss_fn = torch.nn.MSELoss()
			net, final_loss, learning_curve = train_neural_net(
				model,
				loss_fn,
				X=x_train,
				y=y_train,
				n_replicates=n_replicates,
				max_iter=max_iter)
			print('\n\tBest loss: {}\n'.format(final_loss))
			# Determine estimated class labels for test set
			y_pred = net(x_validation)

			# Determine errors and errors
			se_inner = (y_pred.float() - y_validation.float())**2 # squared error
			mse_inner = (sum(se_inner).type(torch.float)/len(y_validation)).data.cpu().numpy() #mean

		y_pred = net(x_evaluation)

		# Determine errors and errors
		se_outer = (y_pred.float() - y_evaluation.float())**2 # squared error
		mse_outer = (sum(se_outer).type(torch.float)/len(y_evaluation)).data.cpu().numpy() #mean
		errors.append(mse_outer)

		# Display the learning curve for the best net in the current fold
		h, = summaries_axes[0].plot(learning_curve, color=color_list[outer_index])
		h.set_label('CV fold {0}'.format(outer_index+1))
		summaries_axes[0].set_xlabel('Iterations')
		summaries_axes[0].set_xlim((0, max_iter))
		summaries_axes[0].set_ylabel('Loss')
		summaries_axes[0].set_title('Learning curves for {} hidden neurons'.format(N_HIDDEN_NEURONS))

	# Display the MSE across outer folds
	summaries_axes[1].bar(np.arange(1, K_1+1), np.squeeze(np.asarray(errors)), color=color_list)
	summaries_axes[1].set_xlabel('Outer fold')
	summaries_axes[1].set_xticks(np.arange(1, K_1+1))
	summaries_axes[1].set_ylabel('MSE')
	summaries_axes[1].set_title('Generalization error across outer folds')
	plt.show()
    
    
    
	# Print the average classification error rate
	print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))

	# When dealing with regression outputs, a simple way of looking at the quality
	# of predictions visually is by plotting the estimated value as a function of 
	# the true/known value - these values should all be along a straight line "y=x", 
	# and if the points are above the line, the model overestimates, whereas if the
	# points are below the y=x line, then the model underestimates the value
	plt.figure(figsize=(10,10))
	y_est = y_pred.data.cpu().numpy(); y_true = y_evaluation.data.cpu().numpy()
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




        #errors.append(mse_inner) # save the best model
                #for epoch in range(epochs):
                    #net = train_model(x_train, y_train, net, optimizer, criterion)
                    # Evaluate the model
                #with torch.no_grad():
                    #y_pred = evaluate_model(x_valid, y_valid, net)
                    #current_mse = mse(y_valid, y_pred)
                    #current_pcc = pcc(y_valid, y_pred)



                        # Check if the current model is the best
                    #if current_mse < best_mse:
                        #best_mse = current_mse
                        #best_pcc = current_pcc
                        #best_model_path = os.path.join(results_dir,"_best_model.pth")
                        #torch.save(net.state_dict(), best_model_path)

    # Write the results to a file
    #results_file_path = os.path.join(results_dir, "_results.txt")
    #with open(results_file_path, "w") as f:
    #    f.write("Best Model: {}\n".format(best_model_path))
    #    f.write("Best MSE: {}\n".format(best_mse))
    #    f.write("Best PCC: {}\n".format(best_pcc))

    #print("\n--------------------------\n")
# %%



