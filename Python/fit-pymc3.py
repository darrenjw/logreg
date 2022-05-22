#!/usr/bin/env python3
# fit-pymc3.py
# Fit using PyMC3

import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats

import pymc3 as pm
import theano
import theano.tensor as tt

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
    eta = tt.dot(X, beta)
    pm.Bernoulli('y', logit_p=eta, observed=y)
    traces = pm.sample(2500, tune=1000, init="adapt_diag", return_inferencedata=True)

print("MCMC finished.")
print(traces.posterior.beta.shape)
out = np.concatenate((traces.posterior.beta[0],
                     traces.posterior.beta[1],
                     traces.posterior.beta[2],
                      traces.posterior.beta[3]))
print(out.shape)
print("Posterior summaries:")
summ = scipy.stats.describe(out)
print(summ)
print("\nMean: " + str(summ.mean))
print("Variance: " + str(summ.variance))

import matplotlib.pyplot as plt
figure, axis = plt.subplots(4, 2)
for i in range(8):
    axis[i // 2, i % 2].plot(range(out.shape[0]), out[:,i])
    axis[i // 2, i % 2].set_title(f'Trace plot for the variable {i}')
plt.savefig("pymc3-nuts-trace.png")
#plt.show()

figure, axis = plt.subplots(4, 2)
for i in range(8):
    axis[i // 2, i % 2].hist(out[:,i], 50)
    axis[i // 2, i % 2].set_title(f'Histogram for variable {i}')
plt.savefig("pymc3-nuts-hist.png")
#plt.show()

figure, axis = plt.subplots(4, 2)
for i in range(8):
    axis[i // 2, i % 2].acorr(out[:,i] - np.mean(out[:,i]), maxlags=100)
    axis[i // 2, i % 2].set_title(f'ACF for variable {i}')
plt.savefig("pymc3-nuts-acf.png")
#plt.show()



# eof
