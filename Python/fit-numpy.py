#!/usr/bin/env python3
# fit-numpy.py
# Basic Bayesian fits using regular numpy

import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_parquet(os.path.join("..", "pima.parquet"))
print(df)
n,p = df.shape
print(n, p)

y = pd.get_dummies(df["type"])["Yes"].to_numpy(dtype='float32')
X = df.drop(columns="type").to_numpy()
X = np.hstack((np.ones((n,1)), X))
print(X)
print(y)

def ll(beta):
    return np.sum(-np.log(1 + np.exp(-(2*y - 1)*(X.dot(beta)))))

init = np.random.randn(p)*0.1
print(init)
print(ll(init))

print("First, MLE:")
res = minimize(lambda x: -ll(x), init, method='BFGS')
print(res.x)
print(ll(res.x))





# eof
