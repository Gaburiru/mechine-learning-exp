from sklearn import datasets as ds
import numpy as np
X_train,y_train=ds.load_svmlight_file('D:/exp2/train.txt',n_features=123)
X_test,y_test=ds.load_svmlight_file('D:/exp2/test.txt',n_features=123)
ones=np.ones((32561,1))
X_train=np.c_[ones,X_train.toarray()]
ones_=np.ones((16281,1))
X_test=np.c_[ones_,X_test.toarray()]

def loss(W,X,y):
    if(y.dot(X.dot(W))<1):
        return 1-y.dot(X.dot(W))
    else:
        return 0
    

def grad(W,X,y):
    if(y.dot(X.dot(W))<1):
        return -1*y.dot(X)
    return 0
    
rgen = np.random.RandomState(1)
W = rgen.normal(loc=0.0, scale=0.01, size= X_train.shape[1])
print('损失函数值loss:',loss(W,X_train[10000:12000],y_train[10000:12000]))
print('梯度:',grad(W,X_train[10000:12000],y_train[10000:12000]))
eta=0.000001
lval_=[]
for n in range(0,100):
    
    for i in range(0,10):
        W=W-eta*grad(W,X_train[i*20:(i+1)*20],y_train[i*20:(i+1)*20])

    Lvalidation=loss(W,X_test,y_test)
    lval_.append(Lvalidation)
    
#print(lval_)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ax[0].plot(range(1, len(lval_) + 1), lval_, marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Lvalidation')
ax[0].set_title('Adaline - Learning rate 0.000001')


