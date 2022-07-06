#!/usr/bin/env python3
# fit-np-hmc.py
# Bayesian fit using numpy for HMC

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

print("MAP:")

pscale = np.array([10.,1.,1.,1.,1.,1.,1.,1.])

def lprior(beta):
    return np.sum(sp.stats.norm.logpdf(beta, loc=0, scale=pscale))

def lpost(beta):
    return ll(beta) + lprior(beta)

def glp(beta):
    glpr = -beta/(pscale*pscale)
    gll = (X.T).dot(y - 1/(1 + np.exp(-X.dot(beta))))
    return (glpr + gll)

def glp(beta):
    glpr = -beta/(pscale*pscale)
    gll = (X.T).dot(y - 1/(1 + np.exp(-X.dot(beta))))
    return (glpr + gll)

res = minimize(lambda x: -lpost(x), init, jac=lambda x: -glp(x), method='BFGS')
print(res.x)
print(ll(res.x))
print(glp(res.x))

print("HMC:")

def mhKernel(lpost, rprop):
    def kernel(x):
        prop = rprop(x)
        a = lpost(prop) - lpost(x)
        if (np.log(np.random.rand()) < a):
            x = prop
        return x
    return kernel
        
def hmcKernel(lpi, glpi, eps = 1e-4, l=10, dmm = 1):
    sdmm = np.sqrt(dmm)
    def leapf(q, p):    
        p = p + 0.5*eps*glpi(q)
        for i in range(l):
            q = q + eps*p/dmm
            if (i < l-1):
                p = p + eps*glpi(q)
            else:
                p = p + 0.5*eps*glpi(q)
        return (q, -p)
    def alpi(x):
        (q, p) = x
        return lpi(q) - 0.5*np.sum((p**2)/dmm)
    def rprop(x):
        (q, p) = x
        return leapf(q, p)
    mhk = mhKernel(alpi, rprop)
    def kern(q):
        d = len(q)
        p = np.random.randn(d)*sdmm
        return mhk((q, p))[0]
    return kern
    
def mcmc(init, kernel, thin = 10, iters = 10000, verb = True):
    p = len(init)
    mat = np.zeros((iters, p))
    x = init
    if (verb):
        print(str(iters) + " iterations")
    for i in range(iters):
        if (verb):
            print(str(i), end=" ", flush=True)
        for j in range(thin):
            x = kernel(x)
        mat[i,:] = x
    if (verb):
        print("\nDone.", flush=True)
    return mat

pre = np.array([100.,1.,1.,1.,1.,1.,25.,1.])

out = mcmc(res.x,
           hmcKernel(lpost, glp, eps=1e-3, l=50, dmm=1/pre), thin=20)

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
plt.savefig("np-hmc-trace.png")
#plt.show()

figure, axis = plt.subplots(4, 2)
for i in range(8):
    axis[i // 2, i % 2].hist(out[:,i], 50)
    axis[i // 2, i % 2].set_title(f'Histogram for variable {i}')
plt.savefig("np-hmc-hist.png")
#plt.show()

figure, axis = plt.subplots(4, 2)
for i in range(8):
    axis[i // 2, i % 2].acorr(out[:,i] - np.mean(out[:,i]), maxlags=100)
    axis[i // 2, i % 2].set_title(f'ACF for variable {i}')
plt.savefig("np-hmc-acf.png")
#plt.show()



# eof
