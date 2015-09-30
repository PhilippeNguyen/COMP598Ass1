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
    sigma2 = SSR / (X.shape[0]-2)
    myLogLik = -0.5 * X.shape[0] * np.log(2*np.pi) - 0.5 * X.shape[0] * np.log(sigma2) - SSR/(2*sigma2)
    return myLogLik

def getSDBetas(Y, X, W):
    sigma2 = getSigma2(Y, X, W)
    toRet = np.sqrt(np.abs(sigma2 / np.dot(X.T, X)))
    return np.diag(toRet)

def getSignificance(Y, X, W):
    SD = getSDBetas(Y, X, W)
    return np.abs(W/SD)

def getAdjustedR2(Y, X, W):
    Ymean = np.mean(Y)
    mySSR = getSSR(Y, X, W)
    #R2 = np.linalg.norm(np.dot(X, W) - Ymean, ord=2)/np.linalg.norm(Y - Ymean, ord=2) 
    #adjustedR2 = 1 - ((1 - R2) * (X.shape[0]-1)/(X.shape[0]-X.shape[1]-1))
    num = 1/(X.shape[0] - X.shape[1]) * np.sum(mySSR)
    denom = 1.0/(X.shape[0]-1) * np.linalg.norm(Y - Ymean, ord = 2)
    return 1-( num/denom )


def getAIC(Y, X, W):
    myLogLik = logLik(Y, X, W)
    return myLogLik - X.shape[1]

def getBIC(Y, X, W):
    myLogLik = logLik(Y, X, W)
    return myLogLik - 0.5 * X.shape[1] * np.log (X.shape[0])
    
    
def getModelInfo(Y, X, W):
    W_significance = getSignificance(Y, X, W)
    r2 = getAdjustedR2(Y, X, W)
    aic = getAIC(Y, X, W)
    bic = getBIC(Y, X, W)
    myLogLik = logLik(Y, X, W)
    toRet = {'Weights': W, 'Significance': W_significance, 'Adjusted R2': r2, 'AIC': aic, 'BIC': bic, 'Log Likelihood': myLogLik}
    return toRet

def gradientDescent(Y, X, tolerance, nb_iterations, learning_rate, ridge = False, penalty_rate = 0):

    epsilon = 9999
    i = 1
    W = np.zeros(X.shape[1]) #np.random.normal(0, 1, X.shape[1])

    while (epsilon > tolerance and i < nb_iterations):
        fitted_values = np.dot(X, W)
        loss = fitted_values - Y
        penalty = 0
        if(ridge == True):
            penalty = 2 * penalty_rate * np.linalg.norm(W, ord = 1)

        gradient = 2 * np.dot(X.T, loss)/X.shape[0] + penalty
        print np.linalg.norm(loss, ord = 1) / X.shape[0], np.max(gradient)
        W_new = W - learning_rate * gradient 
        epsilon = np.linalg.norm(W_new - W, ord = 1) / X.shape[1]
        i += 1
        W = W_new
    toRet = getModelInfo(Y, X, W)
    return toRet

#nInstances = 500
#nParams = 10
#noiseLevel = 0
#trueWeights = np.asarray(range(nParams))
##    trueWeights[5:] = 0
##    trueWeights[:4] = 0
#X = np.random.rand(nInstances,nParams)
#Y = np.dot(X,trueWeights) + noiseLevel*np.random.rand(np.size(nInstances))
#Yfitted = np.dot(X,test['Weights'])
#
#test = gradientDescent(Y, X, 0, 10000, 0.01)
#
