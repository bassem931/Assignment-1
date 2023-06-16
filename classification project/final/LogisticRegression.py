import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    #constructing the classifier learning rate is lr, regularization parameter is alpha
    def __init__(self, d=1, lr=0.01, alpha=2.56, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.d = d
        self.weights = None
        self.alpha = alpha

    #the training process using the sigmoid function and gradient descent
    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_samples, n_classes = y.shape
        self.weights = np.random.randn(self.d,n_features,n_classes)
        for _ in range(self.n_iters):
            linear_pred = np.zeros((np.dot(X, self.weights[0])).shape)
            for d in range(self.d):
                linear_pred += np.dot(X**(d+1), self.weights[d])
            predictions = sigmoid(linear_pred)
            dw = np.dot((X.T)**(d+1), (predictions - y))
            self.weights = self.weights - self.lr*(1/n_samples) * (dw+(self.alpha*self.weights))

    #the testing process
    def predict(self, X):
        linear_pred = np.zeros((np.dot(X, self.weights[0])).shape)
        for d in range(self.d):
            linear_pred += np.dot(X**(d+1), self.weights[d])
        y_pred = sigmoid(linear_pred)
        n_samples, n_features = X.shape
        class_pred = np.zeros((n_samples,self.weights.shape[2]))
        for i in range(len(y_pred)):
            for j in range(len(y_pred[i])):
                if (y_pred[i][j]<0.5):
                    class_pred[i][j] = 0
                else:
                    class_pred[i][j] = 1
                        
        return class_pred