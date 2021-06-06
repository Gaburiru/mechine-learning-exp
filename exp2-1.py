# -*- coding: utf-8 -*-
import numpy as np
class LogisticRegressionGD(object):
    """Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size= X.shape[1])
        self.cost_ = []
        self.grad_=[]
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * X.T.dot(errors) 
            self.grad_.append(X.T.dot(errors))
            # note that we compute the logistic `cost` now
            # instead of the sum of squared errors cost
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_)

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
from sklearn import datasets as ds
X_train,y_train=ds.load_svmlight_file('D:/exp2/train.txt',n_features=123)
X_test,y_test=ds.load_svmlight_file('D:/exp2/test.txt',n_features=123)
ones=np.ones((32561,1))
X_train=np.c_[ones,X_train.toarray()]
ones_=np.ones((16281,1))
X_test=np.c_[ones_,X_test.toarray()]
X_train_01_subset = X_train[(y_train == -1) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == -1) | (y_train == 1)]
test=LogisticRegressionGD(eta=0.0000001, n_iter=1, random_state=1)
test.fit(X_train_01_subset[0:50],y_train_01_subset[0:50])
print('梯度:',test.grad_)
lrgd = LogisticRegressionGD(eta=0.0000001, n_iter=6, random_state=1)
lrgd.fit(X_train_01_subset,
    y_train_01_subset)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ax[0].plot(range(1, len(lrgd.cost_) + 1), np.log10(lrgd.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Lvalidation')
ax[0].set_title('Adaline - Learning rate 0.000001')

lrgd.fit(X_test,y_test)
ax[0].plot(range(1, len(lrgd.cost_) + 1),np.log10(lrgd.cost_), marker='o')

        
    