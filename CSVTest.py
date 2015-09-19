# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:00:08 2015

@author: Philippe
"""

import numpy as np
import csv
from OLSClosed import OLSClosed

    

if __name__ == "__main__":
    
    kFolds = 5
    
    
    xList = []
    yList = []
    
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
    
#    Perform Cross Validation for K-Folds
    padding = (-len(X))%kFolds
    np.split(np.concatenate((a,np.zeros(padding))),kFolds)
    
    aa = split_padded(X,kFolds)
    
    wEst = OLSClosed(X,Y)
    
    