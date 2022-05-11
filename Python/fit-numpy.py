#!/usr/bin/env python3
# fit-numpy.py
# Basic Bayesian fits using regular numpy

import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
from scipy.optimize import minimize

df = pd.read_parquet(os.path.join("..", "pima.parquet"))
print(df)
n, p = df.shape
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

print("MAP next:")

def lprior(beta):
    return (sp.stats.norm.logpdf(beta[0], loc=0, scale=10) + 
            np.sum(sp.stats.norm.logpdf(beta[range(1,p)], loc=0, scale=1)))

print(lprior(init))

def lpost(beta):
    return ll(beta) + lprior(beta)

print(lpost(init))
res = minimize(lambda x: -lpost(x), init, method='BFGS')
print(res.x)
print(ll(res.x))

print("Next, MH:")

def mhKernel(lpost, rprop, dprop = lambda new, old: 1.):
    def kernel(x, ll):
        prop = rprop(x)
        lp = lpost(prop)
        a = lp - ll + dprop(x, prop) - dprop(prop, x)
        if (np.log(np.random.rand()) < a):
            x = prop
            ll = lp
        return x, ll
    return kernel
        
def mcmc(init, kernel, thin = 10, iters = 10000, verb = True):
    p = len(init)
    ll = -np.inf
    mat = np.zeros((iters, p))
    x = init
    if (verb):
        print(str(iters) + " iterations")
    for i in range(iters):
        if (verb):
            print(str(i), end=" ", flush=True)
        for j in range(thin):
            x, ll = kernel(x, ll)
        mat[i,:] = x
    if (verb):
        print("\nDone.", flush=True)
    return mat

pre = np.array([10.,1.,1.,1.,1.,1.,5.,1.])

def rprop(beta):
    return beta + 0.02*pre*np.random.randn(p)

out = mcmc(res.x, mhKernel(lpost, rprop), thin=1000)

print(out)
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
plt.savefig("np-trace.png")
#plt.show()

figure, axis = plt.subplots(4, 2)
for i in range(8):
    axis[i // 2, i % 2].hist(out[:,i], 50)
    axis[i // 2, i % 2].set_title(f'Histogram for variable {i}')
plt.savefig("np-hist.png")
#plt.show()

figure, axis = plt.subplots(4, 2)
for i in range(8):
    axis[i // 2, i % 2].acorr(out[:,i] - np.mean(out[:,i]), maxlags=100)
    axis[i // 2, i % 2].set_title(f'ACF for variable {i}')
plt.savefig("np-acf.png")
#plt.show()



# eof
