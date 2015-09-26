# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:00:08 2015

@author: Philippe
"""

import numpy as np
import csv
from OLSClosed import OLSClosed
from sklearn.utils import shuffle


def mse(yPred,yTrue):
    squareErrors = np.square(yPred-yTrue)
    sse = np.sum(squareErrors)
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
    
    
    #shuffle both X and Y in unison
    X, Y = shuffle(X,Y)
    
    #create and remove test set, 
    #choose the test set arbitrarily
    Xtest = []
    Ytest = []
    X = X
    Y = Y
    
    
    
    
#    Perform Cross Validation for K-Folds
    splitSize = np.size(X,0)/kFolds
    XSplitList = []
    YSplitList = []
    splitStart = 0
    splitEnd = splitSize
    
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
    errorArray = np.zeros(kFolds)
    for i in range(kFolds):
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
        
        #Estimation and prediction using closed form solution        
        wEst = OLSClosed(Xtrain,Ytrain)
        Ypred = np.dot(Xvalid,wEst)
        error = mse(Ypred,Yvalid)
        
        errorArray[i]= error
        
    print 'hello'
        
                
                    
            
        
        
        
        
    

    