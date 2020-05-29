import numpy as np
import scipy
import matplotlib.pyplot as plt 
import seaborn as sns
import random


class KernelRidgeRegression:

    def __init__(self, kernel_fn, reg_lambda=1E-8, kernel_hyperparam = None):
        """
        Constructor
        """
        self.reg_lambda = reg_lambda
        self.alpha = None
        self.kernel_fn = kernel_fn
        self.X_train = None
        self.kernel_hyperparam = kernel_hyperparam
        self.mean = None
        self.std = None

    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-d array
                y is an n-by-1 array
            Returns:
                alpha
        """
        self.mean = np.mean(X, axis = 0)
        self.std = np.std(X, axis = 0)
        X = X - self.mean
        X = X/self.std
        self.X_train = X

        K = self.kernel_fn(X, X, self.kernel_hyperparam)
        self.alpha = np.linalg.solve(K + self.reg_lambda*np.eye(K.shape[0]), y)
        return self.alpha

    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        X = X - self.mean
        X = X/self.std
        K_pred = self.kernel_fn(self.X_train, X, self.kernel_hyperparam)


        # predict
        return K_pred.T.dot(self.alpha)


def true_process(x):
    return 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2)

def generate_data(n):
    x = np.random.rand(n)
    eps = np.random.randn(n)
    y = 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2) + eps
    return x.reshape(-1,1), y

def k_poly(X, Z, d):
    return (1+X.dot(Z.T))**d #for x_i -- row of X and z_i -- row of Z

def squared_l2_dist(X, Z):
    return np.sum((X[:,:,None] - Z[:,:,None].T)**2, axis = 1)

def k_rbf(X, Z, gamma):
    return np.exp(-gamma*squared_l2_dist(X,Z))

def mse(x, y):
    return np.mean((x-y)**2)

def loo_error(model, X, y, reg_lambda, kernel_hyperparam):
    model.reg_lambda = reg_lambda
    model.kernel_hyperparam = kernel_hyperparam
    mse = []
    for i in range(len(X)):
        cv_indices = np.ones(len(X)) #Take all indices
        cv_indices[i] = 0 #except one
        cv_indices = cv_indices.astype('bool')
        X_cv, y_cv = X[cv_indices], y[cv_indices]
        X_eval, y_eval = X[i].reshape(1,-1), y[i].reshape(1,)
        model.fit(X_cv, y_cv)
        y_pred_cur = model.predict(X_eval)
        mse.append(np.mean((y_pred_cur - y_eval)**2))
    score = np.mean(mse)
    return score

def leave_one_out_CV(model, X, y, lambda_range, param_range):
    cv_scores = []
    for reg_lambda in lambda_range:
        for kernel_hyperparam in param_range:
            cv_scores.append([loo_error(model, X, y, reg_lambda, kernel_hyperparam), reg_lambda, kernel_hyperparam])
    return np.array(cv_scores)

def k_fold_cv_error(k, model, X, y, reg_lambda, kernel_hyperparam):
    model.reg_lambda = reg_lambda
    model.kernel_hyperparam = kernel_hyperparam
    idx = np.random.permutation(len(X))
    idx_folds = np.array_split(idx, k) #create random folds
    # if len(idx_folds[-1]) < len(idx_folds[-2]): #don't take the last one if it is too small
    #     idx_folds = idx_folds[:-1]
    # print('Number of folds:', len(idx_folds))
    score = []
    for i in range(len(idx_folds)):
        cur_idx_folds = idx_folds[:]
        val_idx = cur_idx_folds.pop(i)
        train_idx = np.concatenate(cur_idx_folds)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score.append(mse(y_pred, y_val))
    return np.mean(score)

def CV(k_fold, model, X, y, lambda_range, param_range):
    cv_scores = []
    for reg_lambda in lambda_range:
        for kernel_hyperparam in param_range:
            cv_scores.append([k_fold_cv_error(k_fold, model, X, y, reg_lambda, kernel_hyperparam), reg_lambda, kernel_hyperparam])
    return np.array(cv_scores)


def Bootstrap_f(model, X, y, bootstrap_iter = 300):
    X_plot = np.linspace(0,1,100).reshape(-1,1) #fine grid
    fbs = []
    print('BOOOTSTRAP PARAMS:', model.reg_lambda, model.kernel_hyperparam)
    for i in range(bootstrap_iter):
        boot_idx = np.random.choice(len(X), size=len(X), replace = True)
        X_boot, y_boot = X[boot_idx], y[boot_idx]
        model.fit(X_boot, y_boot)
        yb_pred = model.predict(X_plot)
        fbs.append(yb_pred)
    CI_lower = np.percentile(fbs, 5, interpolation = 'lower', axis = 0)
    CI_upper = np.percentile(fbs, 95, interpolation = 'higher', axis = 0)
    return CI_lower, CI_upper, fbs

def Bootstrap_exp_error_diff(polyRidge, rbfRidge, X, y, bootstrap_iter = 300):
    exp_err_diffs = []
    for i in range(bootstrap_iter):
        boot_idx = np.random.choice(len(X), size=len(X), replace = True)
        X_boot, y_boot = X[boot_idx], y[boot_idx]
        poly_yb_pred = polyRidge.predict(X_boot)
        rbf_yb_pred = rbfRidge.predict(X_boot)
        exp_err_diffs.append(np.mean((y_boot - poly_yb_pred)**2 - (y_boot - rbf_yb_pred)**2))
    CI_lower = np.percentile(exp_err_diffs, 5)
    CI_upper = np.percentile(exp_err_diffs, 95)
    return CI_lower, CI_upper

        
if __name__ == "__main__":
    def parts_abc(n, k_fold, names, bootstrap_iter):
        sns.set()
        #parts a and b
        np.random.seed(19)
        X, y = generate_data(n)

        #Find the best params for poly kernel
        lambda_range = 10.0**(-np.arange(2,10))
        d_range = np.arange(4,20)

        polyRidge = KernelRidgeRegression(k_poly)

        loo_scores_poly = CV(k_fold, polyRidge, X, y, lambda_range, d_range)
        best_index_poly = np.argmin(loo_scores_poly[:,0])

        #Plot the results
        print('Best LOO CV params for Polynomial Kernel:', loo_scores_poly[best_index_poly])

        polyRidge.reg_lambda = loo_scores_poly[best_index_poly][1]
        polyRidge.kernel_hyperparam = loo_scores_poly[best_index_poly][2]
        polyRidge.fit(X, y)
        X_plot = np.linspace(0,1,100).reshape(-1,1)
        y_pred = polyRidge.predict(X_plot)
        args = np.argsort(X[:,0])
        plt.figure(figsize = (15,10))
        plt.plot(X[args,0], y[args], 'o', label = 'data')
        plt.plot(X_plot[:,0], true_process(X_plot[:, 0]), '--', label = 'true_process (f)')
        plt.plot(X_plot[:,0], y_pred, '-o', label = 'predicted (f_hat_poly)')
        plt.title('A3: Best Polynomial Kernel Params: lambda {} d {}'.format(
            loo_scores_poly[best_index_poly][1], loo_scores_poly[best_index_poly][2]))
        plt.xlabel('x')
        plt.ylabel('f')
        plt.legend()
        plt.savefig('figures/A3' + names[0] + '_poly.pdf')
        plt.show()


        #Find the best params for rbf kernel
        lambda_range = 10.0**(-np.arange(2,10))
        gamma_range = (1/np.median(squared_l2_dist(X, X)))*np.linspace(0,2,10)

        rbfRidge = KernelRidgeRegression(k_rbf)

        loo_scores_rbf = CV(k_fold, rbfRidge, X, y, lambda_range, gamma_range)
        best_index_rbf = np.argmin(loo_scores_rbf[:,0])

        #Plot the results
        print('Best LOO CV params for RBF Kernel:', loo_scores_rbf[best_index_rbf])

        rbfRidge.reg_lambda = loo_scores_rbf[best_index_rbf][1]
        rbfRidge.kernel_hyperparam = loo_scores_rbf[best_index_rbf][2]
        rbfRidge.fit(X, y)
        X_plot = np.linspace(0,1,100).reshape(-1,1)
        y_pred = rbfRidge.predict(X_plot)
        args = np.argsort(X[:,0])
        plt.figure(figsize = (15,10))
        plt.plot(X[args,0], y[args], 'o', label = 'data')
        plt.plot(X_plot[:,0], true_process(X_plot[:, 0]), '--', label = 'true_process (f)')
        plt.plot(X_plot[:,0],y_pred, '-o', label = 'predicted (f_hat_rbf)')
        plt.title('A3: Best RBF Kernel Params: lambda {} gamma {}'.format(
            loo_scores_rbf[best_index_rbf][1], loo_scores_rbf[best_index_rbf][2]))
        plt.xlabel('x')
        plt.ylabel('f')
        plt.legend()
        plt.savefig('figures/A3' + names[0] + '_rbf.pdf')
        plt.show()


        #Part c

        #Poly kernel
        poly_CI_lower, poly_CI_upper, poly_fbs = Bootstrap_f(polyRidge, X, y, bootstrap_iter)

        polyRidge.reg_lambda = loo_scores_poly[best_index_poly][1]
        polyRidge.kernel_hyperparam = loo_scores_poly[best_index_poly][2]
        polyRidge.fit(X, y)
        X_plot = np.linspace(0,1,100).reshape(-1,1)
        y_pred = polyRidge.predict(X_plot)
        args = np.argsort(X[:,0])
        start = 0
        plt.figure(figsize = (15,10))
        plt.plot(X[args,0], y[args], 'o', label = 'data', color = 'green')
        plt.plot(X_plot[start:,0], true_process(X_plot[:, 0])[start:], '--', label = 'true_process (f)', color = 'C1')
        plt.plot(X_plot[start:,0], y_pred[start:], '-', label = 'predicted (f_hat_poly)', color = 'C0')
        plt.plot(X_plot[start:,0], poly_CI_lower[start:], '-.', label = 'CI_lower', color = 'C0')
        plt.plot(X_plot[start:,0], poly_CI_upper[start:], '-.', label = 'CI_upper', color = 'C0')
        plt.title('A3: 90% CI for the poly Model with lambda {} d {}'.format(
            loo_scores_poly[best_index_poly][1], loo_scores_poly[best_index_poly][2]))
        plt.xlabel('x')
        plt.ylabel('f')
        plt.fill_between(X_plot[start:,0], poly_CI_lower[start:], poly_CI_upper[start:], alpha = 0.2, color = 'C0')
        plt.legend()
        plt.savefig('figures/A3' + names[1] + '_poly.pdf')
        plt.show()

        start = 0
        end = None
        plt.figure(figsize = (15,10))
        plt.plot(X[args,0], y[args], 'o', label = 'data', color = 'green')
        plt.plot(X_plot[start:end, :], true_process(X_plot[:, 0])[start:end], '--', label = 'true_process (f)', color = 'C1')
        plt.plot(X_plot[start:end, :], y_pred[start:end], '-', label = 'predicted (f_hat_poly)', color = 'C0')
        plt.plot(X_plot[start:end, :], poly_CI_lower[start:end], '-.', label = 'CI_lower', color = 'C0')
        plt.plot(X_plot[start:end, :], poly_CI_upper[start:end], '-.', label = 'CI_upper', color = 'C0')
        plt.title('A3: 90% CI for the poly Model with lambda {} d {} (omitting several boundary points)'.format(
            loo_scores_poly[best_index_poly][1], loo_scores_poly[best_index_poly][2]))
        plt.xlabel('x')
        plt.ylabel('f')
        plt.ylim((-10, 10))
        plt.fill_between(X_plot[start:end,0], poly_CI_lower[start:end], poly_CI_upper[start:end], alpha = 0.2, color = 'C0')
        plt.legend()
        plt.savefig('figures/A3' + names[1] + '_poly_zoomed.pdf')
        plt.show()
        
        #Rbf kernel:
        rbf_CI_lower, rbf_CI_upper, rbf_fbs = Bootstrap_f(rbfRidge, X, y, bootstrap_iter)

        rbfRidge.reg_lambda = loo_scores_rbf[best_index_rbf][1]
        rbfRidge.kernel_hyperparam = loo_scores_rbf[best_index_rbf][2]
        rbfRidge.fit(X, y)
        X_plot = np.linspace(0,1,100).reshape(-1,1)
        y_pred = rbfRidge.predict(X_plot)
        args = np.argsort(X[:,0])
        start = 0
        plt.figure(figsize = (15,10))
        plt.plot(X[args,0], y[args], 'o', label = 'data', color = 'green')
        plt.plot(X_plot[start:,0], true_process(X_plot[:, 0])[start:], '--', label = 'true_process (f)', color = 'C1')
        plt.plot(X_plot[start:,0], y_pred[start:], '-', label = 'predicted (f_hat_rbf)', color = 'C0')
        plt.plot(X_plot[start:,0], rbf_CI_lower[start:], '-.', label = 'CI_lower', color = 'C0')
        plt.plot(X_plot[start:,0], rbf_CI_upper[start:], '-.', label = 'CI_upper', color = 'C0')
        plt.title('A3: 90% CI for the rbf Model with lambda {} gamma {}'.format(
            loo_scores_rbf[best_index_rbf][1], loo_scores_rbf[best_index_rbf][2]))
        plt.xlabel('x')
        plt.ylabel('f')
        plt.fill_between(X_plot[:,0], rbf_CI_lower[start:], rbf_CI_upper[start:], alpha = 0.2, color = 'C0')
        plt.legend()
        plt.savefig('figures/A3' + names[1] + '_rbf.pdf')
        plt.show()
        return polyRidge, rbfRidge

    #Run parts a,b,c:
    parts_abc(n = 50, k_fold = 50, names = ['b', 'c'], bootstrap_iter = 300)

    #Part d:
    polyRidge, rbfRidge = parts_abc(n = 300, k_fold = 10, names = ['db', 'dc'], bootstrap_iter = 300)

    #Part e:
    X_new, y_new = generate_data(1000) #generate m additional samples
    err_CI_lower, err_CI_upper = Bootstrap_exp_error_diff(polyRidge, rbfRidge, X_new, y_new, bootstrap_iter = 300)

    print('Expected Error Difference 90% CI:', [err_CI_lower, err_CI_upper])
    #Expected Error Difference 90% CI: [0.0375219151004259, 0.12906511083264974]
    #Does not contain zero -> RBF kernel is better!



