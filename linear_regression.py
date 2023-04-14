
''' 
1-  Approximation
    y' = wx + b

2- Cost Function: 

    MSE = J(w,b) = i/N(sum(yi - (wxi + b))**2)

3 - Gradient descent of cost function wrt w and b
    df/dw and df/db 
    
4 - iTEREATIVE METHOD TO GET TO MINIMUM
    Initiallization of weights and bias
    going towards the negative direction of the gradient descent to find the minimum
    learning rate defines how far we go to find minimum to wards the negative direction

5 - Update weights and bias for every iteration

6 - Man Square error for testing performance

'''
import numpy as np

class LinearRegression():
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # implement the gradient descent here
        #init parameters 
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            #approximation: 
            # y' = w.xi +b
            # Update weights and bias
            # w = w - alpha.dw
            # b = b-alpha.db

            y_pred = np.dot(X,self.weights) +self.bias
            dw = (1/n_samples) * np.dot(X.T,(y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr*db



    def predict(self, X):
        y_pred = np.dot(X,self.weights) +self.bias
        return y_pred



import numpy as no
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X,y = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state = 4)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state =1234)

reg = LinearRegression(lr = 0.001)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

def mse (y_pred,y_true):
    return np.mean((y_true-y_pred)**2)

y_pred_line = reg.predict(X)
mse_value = mse(y_pred,y_test)
print(mse_value)
fig = plt.figure(figsize=(6,6))
m1 = plt.scatter(X[:,0], y)
plt.plot(X,y_pred_line, color = 'black')
plt.show()