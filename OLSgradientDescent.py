import numpy as np
import pdb

def getSigma2(Y, X, W):
    SSR = getSSR(Y, X, W)
    return SSR / (X[0]-2)

def getSSR(Y, X, W):
    loss = Y - np.dot(X, W)
    SSR = np.linalg.norm(loss, ord = 2)
    return SSR

def logLik(Y, X, W):
    SSR = getSSR(Y, X, W)
    sigma2 = SSR / (X[0]-2)
    myLogLik = log(1/sqrt(2*np.pi*sigma2)) - 1/(2 * sigma2) * SSR
    return myLogLik

def getSDBetas(Y, X, W):
    sigma2 = getSigma2(Y, X, W)
    return np.sqrt(sigma2 / np.dot(X.T, X))

def getSignificance(Y, X, W):
    SD = getSDBetas(Y, X, W)
    return np.abs(W/SD)

def getAdjustedR2(Y, X, W):
    Ymean = np.mean(Y)
    R2 = np.linalg.norm(np.dot(X, W) - Ymean, ord=2)/np.linalg.norm(Y - Ymean, ord=2) 
    adjustedR2 = 1 - ((1 - R2) * (X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))
    return adjustedR2


def AIC(Y, X, W):
    myLogLik = logLik(Y, X, W)
    return 2 * X.shape[1] - 2 * myLogLik

def BIC(Y, X, W):
    myLogLik = logLik(Y, X, W)
    return X.shape[1] * log (X.shape[0]) - 2 * myLogLik
    
    
def gradientDescent(Y, X, tolerance, nb_iterations, learning_rate, regularization, penalty_rate = 0):

    epsilon = 9999
    i = 1
    W = np.random.normal(0, 1, X.shape[1])

    while (epsilon > tolerance and i < nb_iterations):
        fitted_values = np.dot(X, W)
        loss = fitted_values - Y

        if(regularization == 'ridge'):
            penalty = 2 * penalty_rate * np.linalg.norm(W, ord = 1)

        gradient = 2 * np.dot(X.T, loss) + penalty
        W_new = W - learning_rate * gradient / X.shape[0]
        epsilon = np.linalg.norm(W_new - W, ord = 1)
        i += 1
        W = W_new
    W_significance = getSignificance(Y, X, W)
    r2 = getAdjustedR2(Y, X, W)
    aic = getAIC(Y, X, W)
    bic = getBIC(Y, X, W)
    toRet = {'Weights': W, 'Significance': W_significance, 'Adjusted R2': r2, 'AIC': aic, 'BIC': bic}
    return toRet

