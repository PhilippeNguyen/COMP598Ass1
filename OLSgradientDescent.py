import numpy as np
import pdb

def gradientDescent(Y, X, eps, nb_iter, alpha):

    epsilon = 9999
    i = 1
    W = np.random.normal(0, 1, X.shape[1])

    while (epsilon > eps and i < nb_iter):
        hypothesis = np.dot(X, W)
        loss = hypothesis - Y
        gradient = 2 * np.dot(X.T, loss)
        W_new = W - alpha * gradient / X.shape[0]
        epsilon = np.linalg.norm(W_new - W, ord = 1)
        i += 1
        W = W_new

    return W


nInstances = 1000
nParams = 10
trueWeights = np.asarray(range(nParams))*2
x = np.random.rand(nInstances,nParams)
y = np.dot(x,trueWeights)

test = gradientDescent(y,x,0.01, 1000, 0.01)

