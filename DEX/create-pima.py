#!/usr/bin/env python3
# create-pima.py
# Create the pima dataset in the form of a Dex source file

import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
from scipy.optimize import minimize

df = pd.read_parquet(os.path.join("..", "pima.parquet"))
n, p = df.shape

y = pd.get_dummies(df["type"])["Yes"].to_numpy(dtype='float32')
X = df.drop(columns="type").to_numpy()
X = np.hstack((np.ones((n,1)), X))

print("")
print("y = [ ", end='')
for i in range(len(y)):
    if (i>0):
        print(", ", end='')
    print(y[i], end='')
print(" ]")
print("")
print("x = [ ", end='')
for i in range(X.shape[0]):
    if (i>0):
        print(",")
    print("[ ", end='')
    for j in range(X.shape[1]):
        if (j>0):
            print(", ", end='')
        print(X[i,j], end='')
    print(" ]", end='')
print("]")
print("")

# eof
