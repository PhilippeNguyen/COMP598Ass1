import numpy as np
import csv
from OLSClosed import OLSClosed
from sklearn.utils import shuffle
from sklearn import linear_model
from OLSgradientDescent import gradientDescent
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import linear_model


nInstances = 500
nParams = 100
noiseLevel = 500
trueWeights = np.asarray(range(nParams))
#    trueWeights[5:] = 0
#    trueWeights[:4] = 0
X = np.random.rand(nInstances,nParams)
Y = np.dot(X,trueWeights) + noiseLevel*np.random.rand(np.size(nInstances))

test = gradientDescent(Y, X, 0, 1000, 0.01)


input = pd.read_csv('OnlineNewsPopularity/OnlineNewsPopularity.csv')

inp = input.as_matrix()

X = inp[:,1:60]
Y = inp[:,60]
Y = Y.reshape(Y.shape[0],1)
X, Y = shuffle(X,Y)
pca = PCA(n_components = 3)

pca.fit(X)
print(pca.explained_variance_ratio_) 
X_new = pca.transform(X)
ones = np.ones(X_new.shape[0])
X_new = np.column_stack((ones, X_new))

clf = linear_model.LinearRegression()
clf.fit(X_new[0:30000,], Y[0:30000,])
clf.fit(X, Y)

clf.predict(X[30000:,])

