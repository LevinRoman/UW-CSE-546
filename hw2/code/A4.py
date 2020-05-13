import numpy as np
import scipy
import matplotlib.pyplot as plt

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
		return (np.linalg.norm(X.dot(self.w) + self.b - y))**2 + self.reg_lambda*np.linalg.norm(self.w, ord = 1)

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
				c_k = 2*np.sum(X[:,k]*(y - (self.b + X[:, not_k_cols].dot(self.w[not_k_cols]))), axis = 0)
				self.w[k] = np.float(np.piecewise(c_k, [c_k < -self.reg_lambda, c_k > self.reg_lambda, ], [(c_k+self.reg_lambda)/a_k, (c_k-self.reg_lambda)/a_k, 0]))
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



def generate_synthetic_data(n, d, k, sigma):
	"""
		Generates the synthetic dataset
	Parameters
	----------
	n : float
	d : float
	k : float
	sigma : float
	
	Returns
	-------
	X : np.array of shape (n,d)
	y : np.array of shape (n,)
	w : np.array of shape (d,)
	"""
	#Create true w:
	w = np.arange(d)/k
	w[k+1:] = 0
	
	#Draw X at random:
	X = np.random.normal(loc=0.0, scale=1.0, size=(n,d))
	#Generate y:
	eps = np.random.normal(loc=0.0, scale=sigma, size=(n,))
	y = X.dot(w) + eps
	return X, y, w


if __name__ == "__main__":
	#Generate synthetic data:
	n = 500
	d = 1000
	k = 100
	sigma = 1
	X, y, w_true = generate_synthetic_data(n, d, k, sigma)
	
	nonzeros = []
	tpr = []
	fdr = []
	lambda_max = np.max(np.sum(2*X*(y - np.mean(y))[:, None], axis = 0))
	print(lambda_max)
	lambdas = [lambda_max/(1.5**i) for i in range(30)]
	w_init = None
	for reg_lambda in lambdas:
		model = Lasso(reg_lambda = reg_lambda)
		model.fit(X,y, w_init, delta = 1e-4)
		w_init = np.copy(model.w)
		total_num_of_nonzeros = np.sum(abs(model.w) > 1e-14)
		number_of_incorrect_nonzeros = np.sum(model.w[abs(w_true) <= 1e-14] > 1e-14)
		number_of_correct_nonzeros = np.sum(model.w[abs(w_true) > 1e-14] > 1e-14)
		nonzeros.append(total_num_of_nonzeros)
		fdr.append(number_of_incorrect_nonzeros/total_num_of_nonzeros)
		tpr.append(number_of_correct_nonzeros/k)
		print('Current nonzero number:', np.sum(abs(model.w) > 1e-14))

	#Part a
	plt.figure(figsize = (15,7))
	plt.plot(lambdas, nonzeros, '-o')
	plt.xscale('log')
	plt.title('Plot 1: nonzero count vs lambda')
	plt.xlabel('lambda')
	plt.ylabel('nonzero elements of w')
	plt.savefig('figures/A4a.pdf')
	plt.show()

	#Part b
	plt.figure(figsize = (15,7))
	plt.plot(fdr, tpr, '-o')
	plt.title('Plot 2: tpr vs fdr')
	plt.xlabel('False Discovery Rate')
	plt.ylabel('True Positive Rate')
	plt.savefig('figures/A4b.pdf')
	plt.show()


