import numpy as np
import matplotlib.pyplot as plt
######################
#Problem B1.d, HW1
######################

# def indicator(x, left, right):
# 	return (x > left) and (x<right)

def f_hat(x, Y, m):
    n = Y.shape[0]
    c = np.array([np.mean(Y[j*m:(j+1)*m]) for j in range(n//m)])
    x_interval_idx = x/(m/n) - 0.001
    x_interval_idx = x_interval_idx.astype('int') #Compute the interval idx
    return c[x_interval_idx]

def true_f(x):
    return 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2)

def mse(x, y):
	return np.mean((x-y)**2)

def bias_squared(X, f, m):
	n = X.shape[0]
	f_arr = f(X)
	fbar = np.array([np.mean(f_arr[j*m:(j+1)*m]) for j in range(n//m)])
	bias_squared = []
	for j in range(n//m):
		for i in range(j*m, (j+1)*m):
			bias_squared.append((fbar[j] - f_arr[i])**2)
	return np.mean(bias_squared)

def variance_squared(sigma, m):
	return sigma**2/m


if __name__ == "__main__":
	n = 256
	sigma = 1
	eps = np.random.normal(0, sigma, n)
	X = np.arange(1,n+1)/n
	Y = true_f(X) + eps
	m = 2**np.arange(1,5+1)
	empirical_mse_arr = np.array([mse(f_hat(X, Y, mi),true_f(X)) for mi in m])
	bias_squared_arr = np.array([bias_squared(X, true_f, mi) for mi in m])
	variance_squared_arr = np.array([variance_squared(sigma, mi) for mi in m])
	avg_error_arr = bias_squared_arr + variance_squared_arr
	

	plt.figure(figsize = (30, 15))
	plt.plot(m, empirical_mse_arr, '-o', label = 'average_empirical_error')
	plt.plot(m, bias_squared_arr, '-o', label = 'average_bias_squared')
	plt.plot(m, variance_squared_arr, '-o', label = 'average_variance_squared')
	plt.plot(m, avg_error_arr, '-o', label = 'average_error')
	plt.title('Problem B1.d')
	plt.xlabel('m')
	plt.ylabel('value')
	plt.legend()
	plt.show()
	plt.savefig('b1_d.pdf')




