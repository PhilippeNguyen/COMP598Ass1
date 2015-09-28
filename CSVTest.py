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

def mse(yPred,yTrue):
    squareErrors = np.square(yPred-yTrue)
    sse = np.sum(squareErrors)
    pdb.set_trace()
    return sse/len(squareErrors)

if __name__ == "__main__":
    
    kFolds = 5
    
    
    xList = []
    yList = []
    
    #read the CSV and create X and Y Matrices
    
    with open('OnlineNewsPopularity/OnlineNewsPopularity.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        irow = 0
        for row in spamreader:
            if irow >=1:
                xList.append(row[2:60])
                yList.append(row[60])
            irow= irow+1
            
    X = np.asarray(xList).astype(float)
    Y = np.asarray(yList).astype(float)
    
    
    nInstances = 500
    nParams = 100
    noiseLevel = 500
    trueWeights = np.asarray(range(nParams))
#    trueWeights[5:] = 0
#    trueWeights[:4] = 0
    X = np.random.rand(nInstances,nParams)
    Y = np.dot(X,trueWeights) + noiseLevel*np.random.rand(np.size(nInstances))

    
    
    #TODO, feature normalization and adding the 1s column
    
    
    #shuffle both X and Y in unison
    X, Y = shuffle(X,Y)
    
    #create and remove test set, 
    #choose the test set arbitrarily
    Xtest = []
    Ytest = []
    X = X
    Y = Y

    test = gradientDescent(Y, X, 0.1, 10000, 0.01, 'nothing')
#    Perform Cross Validation for K-Folds
    splitSize = np.size(X,0)/kFolds
    XSplitList = []
    YSplitList = []
    splitStart = 0
    splitEnd = splitSize
<<<<<<< HEAD
    alphaArray = [10,1.0, 0.1,0.001, 0.0001, 0.0]
    
=======
    alphaArray = [1.0, 0.1,0.001, 0.0001, 0.0]

>>>>>>> origin/master
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
    closedErrorArray = np.zeros((kFolds,np.size(alphaArray)))
    lassoErrorArray = np.zeros((kFolds,np.size(alphaArray)))
    gradientErrorArray = np.zeros((kFolds,np.size(alphaArray)))
    
    for i in range(kFolds):
        print "fold " + str(i)
#        pdb.set_trace()
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

        for j in range(np.size(alphaArray)):
            print "alpha " + str(j)

        #estimation using sklearns Lasso regression
            clf = linear_model.Lasso(alpha=alphaArray[j], max_iter = 10000)
            clf.fit(Xtrain,Ytrain)


            Ypred = clf.predict(Xvalid)
            error  = mse(Ypred,Yvalid)
            lassoErrorArray[i,j] = error

        #Estimation and prediction using closed form solution 
            wEst = OLSClosed(Xtrain,Ytrain,L2 = alphaArray[j])
            Ypred = np.dot(Xvalid,wEst)
            error = mse(Ypred,Yvalid)            
            closedErrorArray[i,j]= error

        #Estimation and prediction using gradient descent
            wEst = gradientDescent(Ytrain,Xtrain,0.0001, 10000, 0.01, 'ridge', alphaArray[j])
            Ypred = np.dot(Xvalid,wEst)
            error = mse(Ypred,Yvalid)
            gradientErrorArray[i,j]= error

    aveLassoError = np.average(lassoErrorArray,axis = 0)
    aveClosedError = np.average(closedErrorArray,axis = 0)
    aveGradientError = np.average(gradientErrorArray,axis = 0)
<<<<<<< HEAD
            
    print aveLassoError
    print aveClosedError
    print aveGradientError    
                
                    
            
        
        
        
        
    
=======

    print 'hello'
>>>>>>> origin/master




