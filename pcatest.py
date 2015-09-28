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
pca = PCA(n_components = 5)

pca.fit(X)
print(pca.explained_variance_ratio_) 
X = pca.transform(X)

ones = np.ones(X.shape[1])
X = np.column_stack((ones, X))
test = gradientDescent(Y, X, 0.1, 1000, 0.01)
