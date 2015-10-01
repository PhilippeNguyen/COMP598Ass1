# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:00:08 2015

@author: Philippe
"""

import numpy as np
import csv
from OLSClosed import OLSClosed
from sklearn.utils import shuffle
from sklearn import linear_model
from OLSgradientDescent import gradientDescent
import pdb
from sklearn.decomposition import PCA
import sys

def mse(yPred,yTrue):
    squareErrors = np.square(yPred-yTrue)
    sse = np.sum(squareErrors)
#    pdb.set_trace()
    return sse/len(squareErrors)

if __name__ == "__main__":
    dataSet = sys.argv[1]
    
    kFolds = 3
    xList = []
    yList = []
    

    

    if dataSet == '1K_data.csv':
        nonPredictiveFeatures = 2
    elif dataSet == 'fullDataSet.csv':
        nonPredictiveFeatures = 3
    elif dataSet == 'smallDataSet.csv':
        nonPredictiveFeatures = 3
    elif dataSet == 'OnlineNewsPopularity/OnlineNewsPopularity.csv':   
        nonPredictiveFeatures = 2
    else:
        print 'non Predictive Features for this data set unknown, using 2'
        nonPredictiveFeatures = 3
        
    #read the CSV and create X and Y Matrices
    
    
    with open(dataSet, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        irow = 0
        for row in spamreader:
            if irow >=1:
                numFeatures = len(row)
                xList.append(row[nonPredictiveFeatures:numFeatures -1])
                yList.append(row[numFeatures -1])
            irow= irow+1
            
    X = np.asarray(xList).astype(float)
    Y = np.asarray(yList).astype(float)
    
    
#    uncomment this next block to use synthetic data 
#    nInstances = 500
#    nParams = 100
#    noiseLevel = 0
#    trueWeights = np.asarray(range(nParams))
##    trueWeights[5:] = 0
##    trueWeights[:4] = 0
#    X = np.random.rand(nInstances,nParams)
#    Y = np.dot(X,trueWeights) + noiseLevel*np.random.rand(np.size(nInstances))

    
    
    #feature normalization
    maxXArray = np.max(np.abs(X),axis=0)
    #if all elements in a feature are zero, then the feature is useless
    zeroFeatures = np.where(maxXArray == 0)[0] 
    zeroFeatures.tolist()
    X = np.delete(X,zeroFeatures,1)
    maxXArray= np.delete(maxXArray,zeroFeatures,0)
    
    invMaxXArray = 1/maxXArray
    X = X*invMaxXArray
    
    
    #shuffle both X and Y in unison
    X, Y = shuffle(X,Y,random_state=1)
    
    
        #PCA
    X_old = X
    pca = PCA()
    pca.fit(X)
        #choose number of components which will explain more than 95% of the variance
    sumVariance= np.cumsum(pca.explained_variance_ratio_)
    numComponents = np.argmax(sumVariance>0.95)
    pca = PCA(n_components=numComponents)
    pca.fit(X)
    
    # add 1s column
    X = np.c_[np.ones(np.size(X,axis=0)),X]
    
    #create and remove test set, 
    #choose the test set arbitrarily
    numExamples = np.size(X,axis=0)
    numTestSet = np.round(numExamples/10)
    Xtest = X[:numTestSet]
    Ytest = Y[:numTestSet]
    X = X[numTestSet:]
    Y = Y[numTestSet:]

#    test = gradientDescent(Y, X, 0.1, 10000, 0.01, 'nothing')
#    Perform Cross Validation for K-Folds
    splitSize = np.size(X,0)/kFolds
    XSplitList = []
    YSplitList = []
    splitStart = 0
    splitEnd = splitSize

    lambArray = [0.25, 1e-1, 1e-2, 1e-3,1e-4,1e-5,1e-10]
    

    #divides the dataset into k-lists
    for i in range(kFolds):
        if i != kFolds -1:
            XSplitList.append(X[splitStart:splitEnd])
            YSplitList.append(Y[splitStart:splitEnd])
        else:
            XSplitList.append(X[splitStart:])
            YSplitList.append(Y[splitStart:])
                    
        splitStart = splitStart +splitSize
        splitEnd = splitEnd + splitSize
        
    # for each kFold, set it aside as a validation set, and use the rest as training
    closedErrorArray = np.zeros((kFolds,np.size(lambArray)))
    lassoErrorArray = np.zeros((kFolds,np.size(lambArray)))
    gradientErrorArray = np.zeros((kFolds,np.size(lambArray)))
    pcaErrorArray =  np.zeros((kFolds,np.size(lambArray)))
    
    for i in range(kFolds):
        print "fold " + str(i)

        #go through the and generate a validation and training set
        Xvalid = []
        Xtrain = np.array([], dtype=np.float64).reshape(0,np.size(X,1))
        Yvalid = []
        Ytrain = np.array([], dtype=np.float64).reshape(0)

        for j in range(kFolds):
            if j != i:
                Xtrain = np.vstack((Xtrain,XSplitList[j]))
                Ytrain = np.concatenate((Ytrain,YSplitList[j]))
            else:
                Xvalid = XSplitList[j]
                Yvalid = YSplitList[j]

        #run all hyperparameters

        for j in range(np.size(lambArray)):
            print "alpha " + str(j)

        #estimation using sklearns Lasso regression
            clf = linear_model.Lasso(alpha=lambArray[j], max_iter = 10000)
            clf.fit(Xtrain,Ytrain)


            Ypred = clf.predict(Xvalid)
            error  = mse(Ypred,Yvalid)
            lassoErrorArray[i,j] = error

        #Estimation and prediction using closed form solution 
            wEst = OLSClosed(Xtrain,Ytrain,L2 = lambArray[j])
            Ypred = np.dot(Xvalid,wEst)
            error = mse(Ypred,Yvalid)            
            closedErrorArray[i,j]= error

        #Estimation and prediction using gradient descent
            wEst = gradientDescent(Ytrain,Xtrain,0.0001, 10000, 0.01, 'ridge', lambArray[j])
            Ypred = np.dot(Xvalid,wEst['Weights'])
            error = mse(Ypred,Yvalid)
            gradientErrorArray[i,j]= error
            
        #Estimation and prediction using closed form solution AND PCA
            wEst = OLSClosed(np.hstack((np.ones((np.size(Xtrain,axis=0) ,1)),pca.transform(Xtrain[:,1:]))),Ytrain,L2 = lambArray[j])
            Ypred = np.dot(np.hstack((np.ones((np.size(Xvalid,axis=0) ,1)),pca.transform(Xvalid[:,1:]))),wEst)
            error = mse(Ypred,Yvalid)            
            pcaErrorArray[i,j]= error    

    aveLassoError = np.average(lassoErrorArray,axis = 0)
    aveClosedError = np.average(closedErrorArray,axis = 0)
    aveGradientError = np.average(gradientErrorArray,axis = 0)
    avePCAError = np.average(pcaErrorArray,axis = 0)
    
    minLasso = np.argmin(aveLassoError)
    minClosed = np.argmin(aveClosedError)
    minGradient = np.argmin(aveGradientError)
    minPCA = np.argmin(avePCAError)
    
    np.savetxt('aveLassoError.csv',aveLassoError,delimiter = ',')
    np.savetxt('aveClosedError.csv',aveClosedError,delimiter = ',')
    np.savetxt('aveGradientError.csv',aveGradientError,delimiter = ',')
    np.savetxt('avePCAError.csv',avePCAError,delimiter = ',')
    np.savetxt('lambArray.csv',lambArray,delimiter = ',')
    
    #run on test set
    clf = linear_model.Lasso(alpha=lambArray[minLasso], max_iter = 10000)
    clf.fit(X,Y)
    Ypred = clf.predict(Xtest)
    lassoTestError  = mse(Ypred,Ytest)
    print 'lassoTestError'
    print lassoTestError
    
    wEst = OLSClosed(X,Y,L2 = lambArray[minClosed])
    Ypred = np.dot(Xtest,wEst)
    closedTestError  = mse(Ypred,Ytest)   
    print 'closedTestError'         
    print closedTestError
    
    wEst = gradientDescent(Y,X,0.0001, 10000, 0.01, 'ridge', lambArray[minGradient])
    Ypred = np.dot(Xtest,wEst['Weights'])
    gradientTestError  = mse(Ypred,Ytest)
    print 'gradientTestError'
    print gradientTestError
    
    wEst = OLSClosed(np.hstack((np.ones((np.size(X,axis=0) ,1)),pca.transform(X[:,1:]))),Y,L2 = lambArray[minPCA])
    Ypred = np.dot(np.hstack((np.ones((np.size(Xtest,axis=0) ,1)),pca.transform(Xtest[:,1:]))),wEst)
    pcaTestError  = mse(Ypred,Ytest)   
    print 'pcaTestError'     
    print pcaTestError

            
            


