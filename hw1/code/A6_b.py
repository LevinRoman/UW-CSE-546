import numpy as np
from mnist import MNIST

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, labels_train, X_test, labels_test

def one_hot_encoder(labels_train):
    """
    Performs one-hot encoding of the labels

    Parameters
    ----------
    labels_train : np.array of shape (n,)
        Labels
    
    Returns
    -------
    np.array of shape (n,k)
        One-hot-encoded labels
    """
    return np.eye(len(set(labels_train)))[labels_train]

def accuracy_error(y_true, y_pred):
    """
    Trains ridge regression using closed-form solution

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

def train(X, Y, reg_lambda):
    """
    Trains ridge regression using closed-form solution

    Parameters
    ----------
    X : np.array of shape (n,d)
        Feature matrix
    Y : np.array of shape (n.k)
        One-hot encoded label matrix
    reg_lambda  : float
        Regularization hyper-parameter, >0

    Returns
    -------
    W_hat : np.array of shape (d,k)
        Weight matrix of ridge regression     
    """
    W_hat = np.linalg.solve(X.T.dot(X) + reg_lambda*np.eye(X.shape[1]), X.T.dot(Y))
    return W_hat 

def predict(W, X_prime):
    """
    Return ridge regression predictions

    Parameters
    ----------
    X : np.array of shape (m,d)
        Feature matrix
    W : np.array of shape (d,k)
        Weight matrix of ridge regression
    reg_lambda  : float
        Regularization hyper-parameter, >0

    Returns
    -------
    predictions : np.array of shape (m,)
        Ridge regression predictions    
    """
    predictions = np.argmax(W.T.dot(X_prime.T), axis = 0)
    return predictions

if __name__ == "__main__":
    X_train, labels_train, X_test, labels_test = load_dataset()
    Y_train = one_hot_encoder(labels_train)
    W_hat = train(X_train, Y_train, 1e-4)
    labels_pred = predict(W_hat, X_test)
    labels_pred_train = predict(W_hat, X_train)
    print('Test error:', accuracy_error(labels_pred, labels_test))
    print('Train error:', accuracy_error(labels_pred_train, labels_train))
    #Test error: 0.14659999999999995
    #Train error: 0.14805000000000001