# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 13:52:23 2015

@author: Philippe
"""

import numpy as np
from OLSClosed import OLSClosed
from sklearn.utils import shuffle


if __name__ == "__main__":
    nInstances = 1000
    nParams = 10
    trueWeights = np.asarray(range(nParams))*2
    x = np.random.rand(nInstances,nParams)
    y = np.dot(x,trueWeights)
    #shuffle both X and Y in unison
    x, y = shuffle(x,y)
    
    estimatedWeights = OLSClosed(x,y)
    
    print estimatedWeights