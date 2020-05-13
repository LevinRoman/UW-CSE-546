import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt


def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, labels_train, X_test, labels_test


def accuracy_error(y_true, y_pred):
    """
    Misclass error
    Parameters
    ----------
    y_true : np.array of shape (m,)
        Vector of true labels
    y_pred : np.array of shape (m,)
        Vector of predicted labels
    
    Returns
    -------
    float
        error: 1-accuracy
    """
    return 1-np.mean(y_true == y_pred)

def mu(w, b, X, y):
    return 1/(1+np.exp(-y*(b + X.dot(w))))

def grad_w(w, b, X, y, reg_lambda):
    return np.mean(((mu(w, b, X, y) - 1)*y)[:,None]*X, axis = 0) +2*reg_lambda*w

def grad_b(w, b, X, y):
    return np.mean((mu(w, b, X, y) - 1)*y, axis = 0) 

def J(w, b, X, y, reg_lambda = 0.1):
    return np.mean(np.log(1+np.exp(-y*(b + X.dot(w))))) + reg_lambda*w.dot(w)

def grad_descent(step, X, y, reg_lambda = 0.1, w_init = None, b_init = None, max_iter = 10000):
    n, d = X.shape
    if w_init is None:
        w_init = np.zeros(d)
    if b_init is None:
        b_init = 0
    count = 0
    w = w_init
    b = b_init
    w_prev = w_init + np.inf
    conv_history = []
    w_history = []
    b_history = []
    while np.linalg.norm(w - w_prev, np.inf) >= 1e-4 and count <= max_iter:
        count += 1
        w_prev = np.copy(w)
        w = w - step*grad_w(w, b, X, y, reg_lambda)
        b = b - step*grad_b(w, b, X, y)
        conv_history.append(J(w, b, X, y, reg_lambda))
        w_history.append(w)
        b_history.append(b)
        if count%10 == 0:
            print('Iter ', count, 'Loss: ', conv_history[-1])
    return w, b, conv_history, w_history, b_history

def predict(w, b, X):
    return np.sign(b + X.dot(w))

def SGD(step, batch_size, X, y, reg_lambda = 0.1, w_init = None, b_init = None, max_iter = 10000):
    n, d = X.shape
    if w_init is None:
        w_init = np.zeros(d)
    if b_init is None:
        b_init = 0
    count = 0
    w = w_init
    b = b_init
    w_prev = w_init + np.inf
    conv_history = []
    w_history = []
    b_history = []
    while np.linalg.norm(w - w_prev, np.inf) >= 1e-4 and count <= max_iter:
        #Sample random batch:
        batch_idx = np.random.choice(n, batch_size)
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        
        count += 1
        w_prev = np.copy(w)
        w = w - step*grad_w(w, b, X_batch, y_batch, reg_lambda)
        b = b - step*grad_b(w, b, X_batch, y_batch)
        conv_history.append(J(w, b, X, y, reg_lambda))
        w_history.append(w)
        b_history.append(b)
        if count%10 == 0:
            print('Iter ', count, 'Loss: ', conv_history[-1])
    return w, b, conv_history, w_history, b_history

if __name__ == "__main__":
	X_train_mult, labels_train_mult, X_test_mult, labels_test_mult = load_dataset()
	#take only binary for 2 and 7
	idx_2_7 = (labels_train_mult == 2).astype('int') + (labels_train_mult == 7).astype('int')
	X_train = X_train_mult[idx_2_7.astype('bool')].astype('float')
	y_train = labels_train_mult[idx_2_7.astype('bool')].astype('float')
	y_train[y_train == 7] = 1
	y_train[y_train == 2] = -1
	idx_2_7_test = (labels_test_mult == 2).astype('int') + (labels_test_mult == 7).astype('int')
	X_test = X_test_mult[idx_2_7_test.astype('bool')].astype('float')
	y_test = labels_test_mult[idx_2_7_test.astype('bool')].astype('float')
	y_test[y_test == 7] = 1
	y_test[y_test == 2] = -1

	w, b, conv_history, w_history, b_history = grad_descent(
    0.1, X_train, y_train, reg_lambda = 0.1, w_init = None, b_init = None, max_iter = 10000)
	#Part b1
	train_loss = conv_history
	test_loss = [J(w, b, X_test, y_test, reg_lambda = 0.1) for w, b in zip(w_history, b_history)]

	plt.figure(figsize = (15,10))
	plt.plot(train_loss, '-o', label = 'train_loss')
	plt.plot(test_loss, '-o', label = 'test_loss')
	plt.title('A6.b: Loss')
	plt.legend()
	plt.xlabel('iteration')
	plt.ylabel('error')
	plt.savefig('figures/A6b1.pdf')
	plt.show()

	#Part b2
	y_pred_train = [predict(w, b, X_train) for w, b in zip(w_history, b_history)]
	train_missclass_error = [accuracy_error(y_train, y_pred) for y_pred in y_pred_train]

	y_pred_test = [predict(w, b, X_test) for w, b in zip(w_history, b_history)]
	test_missclass_error = [accuracy_error(y_test, y_pred) for y_pred in y_pred_test]

	plt.figure(figsize = (15,10))
	plt.plot(train_missclass_error, '-o', label = 'train_missclass_error')
	plt.plot(test_missclass_error, '-o', label = 'test_missclass_error')
	plt.title('A6.b: Misclassification Error')
	plt.legend()
	plt.xlabel('iteration')
	plt.ylabel('error')
	plt.savefig('figures/A6b2.pdf')
	plt.show()

	w, b, conv_history, w_history, b_history = SGD(
		step = 0.01, batch_size = 1, X = X_train, y = y_train, max_iter = 500)
	#Part c1
	train_loss = conv_history
	test_loss = [J(w, b, X_test, y_test, reg_lambda = 0.1) for w, b in zip(w_history, b_history)]

	plt.figure(figsize = (15,10))
	plt.plot(train_loss, '-o', label = 'train_loss')
	plt.plot(test_loss, '-o', label = 'test_loss')
	plt.title('A6.c: Loss (SGD, batch_size = 1)')
	plt.legend()
	plt.xlabel('iteration')
	plt.ylabel('error')
	plt.savefig('figures/A6c1.pdf')
	plt.show()

	#Part c2
	y_pred_train = [predict(w, b, X_train) for w, b in zip(w_history, b_history)]
	train_missclass_error = [accuracy_error(y_train, y_pred) for y_pred in y_pred_train]

	y_pred_test = [predict(w, b, X_test) for w, b in zip(w_history, b_history)]
	test_missclass_error = [accuracy_error(y_test, y_pred) for y_pred in y_pred_test]

	plt.figure(figsize = (15,10))
	plt.plot(train_missclass_error, '-o', label = 'train_missclass_error')
	plt.plot(test_missclass_error, '-o', label = 'test_missclass_error')
	plt.title('A6.b: Misclassification Error (SGD, batch_size = 1)')
	plt.legend()
	plt.xlabel('iteration')
	plt.ylabel('error')
	plt.savefig('figures/A6c2.pdf')
	plt.show()

	w, b, conv_history, w_history, b_history = SGD(
    step = 0.01, batch_size = 100, X = X_train, y = y_train, max_iter = 500)
	#Part d1
	train_loss = conv_history
	test_loss = [J(w, b, X_test, y_test, reg_lambda = 0.1) for w, b in zip(w_history, b_history)]

	plt.figure(figsize = (15,10))
	plt.plot(train_loss, '-o', label = 'train_loss')
	plt.plot(test_loss, '-o', label = 'test_loss')
	plt.title('A6.c: Loss (SGD, batch_size = 100)')
	plt.legend()
	plt.xlabel('iteration')
	plt.ylabel('error')
	plt.savefig('figures/A6d1.pdf')
	plt.show()

	#Part c2
	y_pred_train = [predict(w, b, X_train) for w, b in zip(w_history, b_history)]
	train_missclass_error = [accuracy_error(y_train, y_pred) for y_pred in y_pred_train]

	y_pred_test = [predict(w, b, X_test) for w, b in zip(w_history, b_history)]
	test_missclass_error = [accuracy_error(y_test, y_pred) for y_pred in y_pred_test]

	plt.figure(figsize = (15,10))
	plt.plot(train_missclass_error, '-o', label = 'train_missclass_error')
	plt.plot(test_missclass_error, '-o', label = 'test_missclass_error')
	plt.title('A6.b: Misclassification Error (SGD, batch_size = 100)')
	plt.legend()
	plt.xlabel('iteration')
	plt.ylabel('error')
	plt.savefig('figures/A6d2.pdf')
	plt.show()