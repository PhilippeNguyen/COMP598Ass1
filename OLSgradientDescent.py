import numpy as np
import pdb

def gradientDescent(Y, X, eps, nb_iter, alpha, penalty, ll = 0):

    epsilon = 9999
    i = 1
    W = np.random.normal(0, 1, X.shape[1])

    while (epsilon > eps and i < nb_iter):
        hypothesis = np.dot(X, W)
        loss = hypothesis - Y

        if(penalty == 'ridge'):
            penalty = 2 * ll * np.linalg.norm(W, ord = 1)

        gradient = 2 * np.dot(X.T, loss) + penalty
        W_new = W - alpha * gradient / X.shape[0]
        epsilon = np.linalg.norm(W_new - W, ord = 1)
        i += 1
        W = W_new

    return W

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

def AIC(Y, X, W):
    myLogLik = logLik(Y, X, W)
    return 2 * X.shape[1] - 2 * myLogLik

def BIC(Y, X, W):
    myLogLik = logLik(Y, X, W)
    return X.shape[1] * log (X.shape[0]) - 2 * myLogLik

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

def backwardFeatureSelection():
    while(power)


nInstances = 1000
nParams = 10
trueWeights = np.asarray(range(nParams))*2
x = np.random.rand(nInstances,nParams)
y = np.dot(x,trueWeights)

test = gradientDescent(y,x,0.0001, 10000, 0.01, 'ridge', 0.01)

