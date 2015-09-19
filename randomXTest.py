# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 13:52:23 2015

@author: Philippe
"""

import numpy as np
from OLSClosed import OLSClosed


if __name__ == "__main__":
    nInstances = 1000
    nParams = 10
    trueWeights = np.asarray(range(nParams))*2
    x = np.random.rand(nInstances,nParams)
    y = np.dot(x,trueWeights)
    
    
    estimatedWeights = OLSClosed(x,y)
    
    print estimatedWeights