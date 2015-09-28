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
import pandas as pd
from sklearn.decomposition import PCA

input = pd.read_csv('OnlineNewsPopularity/OnlineNewsPopularity.csv')

inp = input.as_matrix()

X = inp[:,1:60]
Y = inp[:,60]
pca = PCA(n_components = 3)

pca.fit(X)
print(pca.explained_variance_ratio_) 
X_new = pca.transform(X)
X_new = X_new 

ones = np.ones(X_new.shape[0])
X_new = np.column_stack((ones, X_new))
test = gradientDescent(Y[0:1000,], X_new[0:1000,], 0, 100000, 1e-12)
