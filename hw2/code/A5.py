######################
#Problem A5, HW2
######################

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd

class Lasso:

	def __init__(self, reg_lambda=1E-8):
		"""
		Constructor
		"""
		self.reg_lambda = reg_lambda
		self.w = None
		self.b = None
		self.conv_history = None

	def objective(self, X, y):
		"""
		Returns Lasso objective value

		Parameters
		----------
		X : np.array of shape (n,d)
			Features
		y : np.array of shape (n,)
			Labels
		Returns
		-------
		float
			Objective value for current w, b and given reg_lambda
		"""
		return (np.linalg.norm(X.dot(self.w) + self.b 
			- y))**2 + self.reg_lambda*np.linalg.norm(self.w, ord = 1)

	def fit(self, X, y, w_init = None, delta = 1e-4):
		"""
			Trains the Lasso model
		Parameters
		----------
		X : np.array of shape (n,d)
			Features
		y : np.array of shape (n,)
			Labels
		w_init : np.array of shape (d,)
			Initial guess for w
		delta : float
			Stopping criterion
		
		Returns
		-------
		convergence_history : array
			convergence history
		"""
		iter_count = 0
		n, d = X.shape
		if w_init is None:
			w_init = np.zeros(d)
		w_prev = w_init + np.inf #just to enter the loop
		self.w = w_init
		convergence_history = []
		a = 2*np.sum(X**2, axis = 0) #precompute a
		print('shape a:', a.shape, 'should be ', d)
		while np.linalg.norm(self.w-w_prev, ord = np.inf) >= delta:
			iter_count += 1
			w_prev = np.copy(self.w)
			self.b = np.mean(y - X.dot(self.w))
			for k in range(d):
				not_k_cols = np.arange(d) != k
				a_k = a[k]
				c_k = 2*np.sum(X[:,k]*(y - (self.b +
					X[:, not_k_cols].dot(self.w[not_k_cols]))), axis = 0)
				self.w[k] = np.float(np.piecewise(c_k,
					[c_k < -self.reg_lambda, c_k > self.reg_lambda, ],
					[(c_k+self.reg_lambda)/a_k, (c_k-self.reg_lambda)/a_k, 0]))
			if iter_count % 1 == 0:
				print('Iter ', iter_count, ' Loss:', self.objective(X,y))
			convergence_history.append(self.objective(X,y))
		self.conv_history = convergence_history
		print('converged in: ', len(convergence_history))
		return convergence_history


	def predict(self, X):
		"""
		Use the trained model to predict values for each instance in X
		Arguments:
			X is a n-by-d numpy array
		Returns:
			an n-by-1 numpy array of the predictions
		"""
		return X.dot(self.w) + self.b

def mse(x, y):
    return np.mean((x-y)**2)

if __name__ == "__main__":
	df_train = pd.read_table('data/crime-train.txt')
	df_test = pd.read_table('data/crime-test.txt')

	y_train = df_train['ViolentCrimesPerPop']
	X_train = df_train.drop('ViolentCrimesPerPop', axis = 1)
	y_test = df_test['ViolentCrimesPerPop']
	X_test = df_test.drop('ViolentCrimesPerPop', axis = 1)

	nonzeros = []
	w_regularization_path = []
	train_mse = []
	test_mse = []
	lambda_max = np.max(np.sum(2*X_train.values*(y_train.values - 
		np.mean(y_train.values))[:, None], axis = 0))
	lambdas = [lambda_max/(2**i) for i in range(17)]

	w_init = None
	for reg_lambda in lambdas:
	    model = Lasso(reg_lambda = reg_lambda)
	    model.fit(X_train.values,y_train.values, w_init, delta = 1e-4)
	    w_init = np.copy(model.w) #initialize with the previous solution, this is even faster and the problem is cvx anyway
	    w_regularization_path.append(np.copy(model.w))
	    total_num_of_nonzeros = np.sum(abs(model.w) > 1e-14)
	    nonzeros.append(total_num_of_nonzeros)
	    train_mse.append(mse(model.predict(X_train), y_train))
	    test_mse.append(mse(model.predict(X_test), y_test))
	    print('Current nonzero number:', np.sum(abs(model.w) > 1e-14))

	#Part a
	plt.figure(figsize = (15,10))
	plt.plot(lambdas, nonzeros, '-o')
	plt.xscale('log')
	plt.title('A5.a: Nonzero count vs lambda')
	plt.xlabel('lambda')
	plt.ylabel('# of nonzero elements of w')
	plt.savefig('figures/A5a.pdf')
	plt.show()

	#Part b
	plt.figure(figsize = (15,10))
	w_regularization_path = np.array(w_regularization_path)
	coeffs_names = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
	coeffs_indices = [X_train.columns.get_loc(i) for i in coeffs_names]
	for coeff_path, label in zip(w_regularization_path[:, coeffs_indices].T, coeffs_names):
	    plt.plot(lambdas, coeff_path, '-o', label=label,)
	plt.legend()
	plt.xscale('log')
	plt.title('A5.b: Regularization paths')
	plt.xlabel('lambda')
	plt.ylabel('Coefficient')
	plt.savefig('figures/A5b.pdf')

	#Part c
	plt.figure(figsize = (15,10))
	plt.plot(lambdas, train_mse, '-o', label = 'train_mse')
	plt.plot(lambdas, test_mse, '-o', label = 'test_mse')
	plt.xscale('log')
	plt.title('A5.c: MSE')
	plt.legend()
	plt.xlabel('lambda')
	plt.ylabel('error')
	plt.savefig('figures/A5c.pdf')
	plt.show()

	#Part d
	model = Lasso(reg_lambda = 30)
	model.fit(X_train.values,y_train.values, w_init, delta = 1e-4)
	plt.figure(figsize = (15,10))
	plt.plot(model.w, '-o')
	plt.title('A5.d: Weights')
	plt.xlabel('Feature index')
	plt.ylabel('Weight')
	plt.savefig('figures/A5d.pdf')
	plt.show()

	print('Largest positive weight:', X_train.columns[np.argmax(model.w)])
	print('Largest negative weight:', X_train.columns[np.argmin(model.w)])