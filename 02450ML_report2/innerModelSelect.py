import numpy as np
import sklearn.linear_model as lm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

#%% 
def rlr_validate_logistic(X, y, lambdas, cvf=10):
    ''' Validate regularized logistic regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation loss) from 'lambdas' list.
        The loss function computed as log loss on the validation set.
        
        Parameters:
        X       training data set
        y       vector of labels
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds     
        
        Returns:
        opt_lambda          value of optimal lambda
    '''
    CV = model_selection.KFold(cvf, shuffle=True)
    train_loss = np.empty((cvf, len(lambdas)))
    val_loss = np.empty((cvf, len(lambdas)))
    f = 0 # no. of folds
    y = y.squeeze()
    
    for train_index, val_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]
         
        # Standardize the training and validation sets based on training set moments
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)
        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_val[:, 1:] = (X_val[:, 1:] - mu) / sigma
        
        # Iteration every lambda
        for l in range(0,len(lambdas)): 
            
            # Fit logistic regression model
            model = lm.LogisticRegression(penalty='l2', C=1.0 / lambdas[l], 
                                          max_iter = 1000, solver = 'newton-cholesky')
            model = model.fit(X_train,y_train)
            
            # Evaluate training and test performance
            train_loss[f,l] = 1 - model.score(X_train, y_train)
            val_loss[f,l] = 1 - model.score(X_val, y_val)
     
        f = f + 1

    opt_lambda = lambdas[np.argmin(np.mean(val_loss, axis=0))]
    
    return opt_lambda

#%% KNN model selection (LeaveOneOut)
def KNNCV(X, y, K=10):
    ''' Validate k-nearest neighbors model using LeaveOneOut validation.
        Find the optimal k (minimizing validation loss) from 1 to 10(default).
        
        Parameters:
        X       training data set
        y       vector of labels
        K       default number of max k     
        
        Returns:
        opt_k    value of optimal k
    '''
    N,_ = X.shape # No. of samples

    CV = model_selection.LeaveOneOut()
    errors = np.zeros((N,K))
    i=0
    for train_index, test_index in CV.split(X, y):    
    
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]

        # Standardization
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)
        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
        
        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1,K+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(X_train, y_train);
            y_est = knclassifier.predict(X_test);
            errors[i,l-1] = np.sum(y_est[0]!=y_test[0])

        i+=1

    return np.argmin(np.mean(errors, axis=0)) + 1