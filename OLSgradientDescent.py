import numpy as np
import pdb

def gradientDescent(Y, X, eps, nb_iter, alpha):

    epsilon = 9999
    i = 1
    W = np.random.normal(0, 1, X.shape[1])
    while (epsilon > eps and i < nb_iter):
        W_new = W - 2 * alpha * (np.dot(np.dot(X.T, X), W) - np.dot(X.T, Y))
        epsilon = np.linalg.norm(W_new - W, ord = 1)
        i += 1
        W = W_new

    return W


