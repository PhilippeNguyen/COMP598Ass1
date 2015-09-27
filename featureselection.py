import numpy as np
import pandas as pd
from sklearn import linear_model

xx = pd.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")

clf = linear_model.Lasso(alpha=2.5, max_iter = 10000)

clf.fit(xx.iloc[:,1:60], xx.iloc[:,60], )

print(clf.coef_)

