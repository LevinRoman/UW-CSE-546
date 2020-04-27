'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree=1, reg_lambda=1E-8):
        """
        Constructor
        """
        self.regLambda = reg_lambda
        self.degree = degree

    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        X_ = X
        for d in range(1, self.degree):
        	X_ = np.c_[X_, X**d]

        return X_


    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """
        n = X.shape[0]
        # polynomial expansion 
        X_ = self.polyfeatures(X, self.degree)
        # normalization (no normalization if n = 1)
        if n!=1:
        	self.mean = X_.mean(axis = 0)
        	self.std = X_.std(axis = 0)
        else:
        	self.mean = 0
        	self.std = 1

        X_ = (X_ - self.mean)/self.std
        # adding ones
        X_ = np.c_[np.ones([n,1]), X_]

        # closed form of regression
        _, d = X_.shape
        # construct reg matrix
        reg_matrix = self.regLambda * np.eye(d)
        reg_matrix[0, 0] = 0

	# analytical solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.pinv(X_.T.dot(X_) + reg_matrix).dot(X_.T).dot(y)


    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        n = X.shape[0]
        X_ = self.polyfeatures(X, self.degree)
        # normalize
        X_ = (X_ - self.mean)/self.std
        # adding ones
        X_ = np.c_[np.ones([n,1]), X_]

        # predict
        return X_.dot(self.theta)


#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    # Here we start from two observations since we do not care about 
    # the performance of the model trained with only 1 data point. 

    for i in range(1,n):
    	Xtrain_i = Xtrain[0:i+1]
    	Ytrain_i = Ytrain[0:i+1]
    	Ytrain_i = Ytrain[0:i+1]

    	# training the model 
    	model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
    	model.fit(Xtrain_i, Ytrain_i)

    	# make predictions
    	predictions_train_i = model.predict(Xtrain_i)
    	predictions_test_i = model.predict(Xtest)
    	# compute errors
    	error_train_i = ((predictions_train_i -Ytrain_i)**2).mean()
    	error_test_i = ((predictions_test_i -Ytest)**2).mean()
    	errorTrain[i] = error_train_i
    	errorTest[i] = error_test_i

    return errorTrain, errorTest
