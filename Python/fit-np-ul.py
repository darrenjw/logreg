#!/usr/bin/env python3
# fit-np-ul.py
# Bayesian fit using numpy for an unadjusted Langevin algorithm (approximate)

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

print("without gradients")
print(lpost(init))
res = minimize(lambda x: -lpost(x), init, method='BFGS')
print(res.x)
print(ll(res.x))

def glp(beta):
    glpr = -beta/(pscale*pscale)
    gll = (X.T).dot(y - 1/(1 + np.exp(-X.dot(beta))))
    return (glpr + gll)

print(glp(init))
print(glp(res.x))

print("with gradients")
res = minimize(lambda x: -lpost(x), init, jac=lambda x: -glp(x), method='BFGS')
print(res.x)
print(ll(res.x))
print(glp(res.x))

print("Unadjusted Langevin (approximate):")

def ulKernel(lpi, glpi, dt = 1e-4, pre = 1):
    p = len(init)
    sdt = np.sqrt(dt)
    spre = np.sqrt(pre)
    advance = lambda x: x + 0.5*pre*glpi(x)*dt
    def kernel(x):
        return advance(x) + np.random.randn(p)*spre*sdt
    return kernel

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

out = mcmc(res.x, ulKernel(lpost, glp, dt=1e-6, pre=pre), thin=2000)

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
plt.savefig("np-ul-trace.png")
#plt.show()

figure, axis = plt.subplots(4, 2)
for i in range(8):
    axis[i // 2, i % 2].hist(out[:,i], 50)
    axis[i // 2, i % 2].set_title(f'Histogram for variable {i}')
plt.savefig("np-ul-hist.png")
#plt.show()

figure, axis = plt.subplots(4, 2)
for i in range(8):
    axis[i // 2, i % 2].acorr(out[:,i] - np.mean(out[:,i]), maxlags=100)
    axis[i // 2, i % 2].set_title(f'ACF for variable {i}')
plt.savefig("np-ul-acf.png")
#plt.show()



# eof
