# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 13:48:06 2015

@author: Philippe
"""
def OLSClosed(x,y, L2=0):
    
    import numpy as np
    insideInverse = np.dot(np.transpose(x),x) + L2*np.eye(np.size(x,1));
    xCM = np.linalg.inv(insideInverse)
    XXX = np.dot(xCM,np.transpose(x))
    wEst = np.dot(XXX,y)
    return wEst
    

