# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 13:48:06 2015

@author: Philippe
"""
def OLSClosed(x,y):
    
    import numpy as np
    
    xCM = xCM = np.linalg.inv(np.dot(np.transpose(x),x))
    XXX = np.dot(xCM,np.transpose(x))
    wEst = np.dot(XXX,y)
    return wEst
    

