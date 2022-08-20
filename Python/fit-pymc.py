#!/usr/bin/env python3
# fit-pymc.py
# Fit using PyMC (3), version >= 4.0

import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats

import pymc as pm
import pymc.math as pmm

df = pd.read_parquet(os.path.join("..", "pima.parquet"))
print(df)
n, p = df.shape
print(n, p)

y = pd.get_dummies(df["type"])["Yes"].to_numpy(dtype='float32')
X = df.drop(columns="type").to_numpy()
X = np.hstack((np.ones((n,1)), X))
print(X)
print(y)

# Now specify model in PyMC3
pscale = np.array([10.,1.,1.,1.,1.,1.,1.,1.])
with pm.Model() as model:
    beta = pm.Normal('beta', 0, pscale, shape=p)
    eta = pmm.matrix_dot(X, beta)
    pm.Bernoulli('y', logit_p=eta, observed=y)
    traces = pm.sample(2500, tune=1000, chains=4, init="adapt_diag", return_inferencedata=True)

print("MCMC finished.")
print(traces.posterior.beta.shape)
out = np.concatenate((traces.posterior.beta[0],
                     traces.posterior.beta[1],
                     traces.posterior.beta[2],
                      traces.posterior.beta[3]))
print(out.shape)
odf = pd.DataFrame(out, columns=["b0","b1","b2","b3","b4","b5","b6","b7"])
odf.to_parquet("fit-pymc.parquet")
print("Posterior summaries:")
summ = scipy.stats.describe(out)
print(summ)
print("\nMean: " + str(summ.mean))
print("Variance: " + str(summ.variance))


# eof
