import numpy as np
import scipy
import matplotlib.pyplot as plt 
import seaborn as sns
import random
from mnist import MNIST

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, labels_train, X_test, labels_test


#Part a
class K_Means:

    def __init__(self, k):
        """
        Constructor
        """
        self.k = k
        self.centroids = None
        self.clusters_train = None
        self.objective_hist = []

    def fit(self, X, max_iter = 10000, eps = 1e-2):
        """
            Finds the K-means centroids using Lloyd's algorithm
            Arguments:
                X is a n-by-d array
            Returns:
        """
        #Initialize centroids
        init_centroids_indices = np.random.choice(len(X), size=self.k, replace = False)
        init_centroids = X[init_centroids_indices]
        # init_centroids = np.random.normal(0.5, 0.5,init_centroids.shape).astype('float32')
        # init_centroids = 10+np.random.randn(self.k, X.shape[1]).astype('float32')

        centroids = np.copy(init_centroids)
        centroids_prev = init_centroids + np.inf

        dist = np.zeros((len(X),self.k))

        count = 0

        while np.linalg.norm(centroids - centroids_prev) > eps and count < max_iter: 
            count += 1
            if count % 10 == 0:
                print('Iter ', count, 'Obj:', obj_cur)
            centroids_prev = np.copy(centroids)
            
            #Find clusters
            for i in range(self.k):
                dist[:,i] = np.linalg.norm(X - centroids[i], axis=1)**2

            cur_clusters = np.argmin(dist, axis = 1)
            assert len(cur_clusters) == len(X)
            
            #recompute centroids
            new_centroids = []
            obj_cur = 0
            for i in range(self.k):
                cluster = X[cur_clusters == i]
                obj_cur += np.sum(np.linalg.norm(cluster - centroids[i], axis = 1)**2)
                centroid = np.mean(cluster, axis = 0)
                new_centroids.append(centroid)
            centroids = np.copy(np.array(new_centroids))
            self.objective_hist.append(obj_cur)

        self.centroids = centroids
        self.clusters_train = cur_clusters
        return 


    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        dist = np.zeros((len(X),self.k))
        for i in range(self.k):
                dist[:,i] = np.linalg.norm(X - self.centroids[i], axis=1)**2
        cur_clusters = np.argmin(dist, axis = 1) #labels
        predict_using_centroids = self.centroids[cur_clusters] #return centroids as predictions
        return predict_using_centroids



if __name__ == "__main__":
    #Part b
    X_train, y_train, X_test, y_test = load_dataset()

    km = K_Means(k = 10)

    km.fit(X_train, max_iter = 100)
    np.save('centroids.npy', km.centroids)

    plt.figure(figsize = (15,10))
    plt.plot(km.objective_hist, '-o')
    plt.title('A4b: K-Means Objective')
    plt.xlabel('iteration')
    plt.ylabel('objective')
    plt.savefig('figures/A4b_obj.pdf')
    plt.show()

    num_row = 2
    num_col = 5
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(km.centroids[i].reshape(28,28), cmap='gray')
        ax.set_title('Centroid {}'.format(i))
    plt.tight_layout()
    plt.savefig('figures/A4b_centroids.pdf')
    plt.show()

    #Part c
    k_range = 2**np.arange(1,7)

    train_err = []
    test_err = []
    for k in k_range:
        print('Current k:', k)
        cur_km = K_Means(k = k)
        cur_km.fit(X_train, max_iter = 40, eps = 1e-1)
        train_prediction = cur_km.predict(X_train)
        train_err.append(np.mean(np.linalg.norm(X_train - train_prediction, axis = 1)**2))
        test_prediction = cur_km.predict(X_test)
        test_err.append(np.mean(np.linalg.norm(X_test - test_prediction, axis = 1)**2))
    
    plt.figure(figsize = (15,10))
    plt.plot(k_range, train_err, '-o', label = 'train_error')
    plt.plot(k_range, test_err, '-o', label = 'test_error')
    plt.title('A4c: K-Means Objective')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('error')
    plt.savefig('figures/A4c_.pdf')
    plt.show()




