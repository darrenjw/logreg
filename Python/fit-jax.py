#!/usr/bin/env python3
# fit-jax.py
# Basic Bayesian fits using JAX

import os
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
from scipy.optimize import minimize

import jax
from jax import grad, jit
import jax.numpy as jnp
import jax.scipy as jsp

df = pd.read_parquet(os.path.join("..", "pima.parquet"))
print(df)
n, p = df.shape
print(n, p)

y = pd.get_dummies(df["type"])["Yes"].to_numpy(dtype='float32')
X = df.drop(columns="type").to_numpy()
X = np.hstack((np.ones((n,1)), X))
print(X)
print(y)

# Now do computations in JAX

X = X.astype(jnp.float32)
y = y.astype(jnp.float32)

@jit
def ll(beta):
    return jnp.sum(-jnp.log(1 + jnp.exp(-(2*y - 1)*jnp.dot(X, beta))))

np.random.seed(41) # for reproducibility
init = np.random.randn(p)*0.1
print(init)
init = init.astype(jnp.float32)
print(ll(init))

print("MAP:")

@jit
def lprior(beta):
    return (jsp.stats.norm.logpdf(beta[0], loc=0, scale=10) + 
            jnp.sum(jsp.stats.norm.logpdf(beta[jnp.array(range(1,p))], loc=0, scale=1)))

@jit
def lpost(beta):
    return ll(beta) + lprior(beta)

print(lpost(init))

glp = jit(grad(lpost))
print(glp(init))

from jax import jacfwd, jacrev
def hessian(f):
    return jacfwd(jacrev(f))

hess = hessian(lpost)
beta = init

# Newton method (log reg is convex)
for i in range(500):
    g = glp(beta)
    step = -jsp.linalg.solve(hess(beta), g)
    for j in range(15):
        if (lpost(beta+step) > lpost(beta)):
            break
        else:
            step = step/2
    beta += step            
    if (jnp.linalg.norm(g) < 0.01):
        break
    
print(beta)
print(ll(beta))
print(jnp.linalg.norm(glp(beta)))

print("Next, MH:")



def metHast(init, lpost, rprop, dprop = lambda new, old: 1.,
           thin = 10, iters = 10000, verb = True):
    p = len(init)
    olp = -np.inf
    mat = np.zeros((iters, p))
    x = init
    if (verb):
        print(str(iters) + " iterations")
    for i in range(iters):
        if (verb):
            print(str(i), end=" ", flush=True)
        for j in range(thin):
            prop = rprop(x)
            lp = lpost(prop)
            a = lp - olp + dprop(x, prop) - dprop(prop, x)
            if (np.log(np.random.rand()) < a):
                x = prop
                olp = lp
        mat[i,:] = x
    if (verb):
        print("\nDone.", flush=True)
    return mat

pre = jnp.array([10.,1.,1.,1.,1.,1.,5.,1.])

def rprop(beta):
    return beta + 0.02*pre*np.random.randn(p)

out = metHast(init, lpost, rprop, thin=100)

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
plt.savefig("jax-trace.png")
#plt.show()

figure, axis = plt.subplots(4, 2)
for i in range(8):
    axis[i // 2, i % 2].hist(out[:,i], 50)
    axis[i // 2, i % 2].set_title(f'Histogram for variable {i}')
plt.savefig("jax-hist.png")
#plt.show()

figure, axis = plt.subplots(4, 2)
for i in range(8):
    axis[i // 2, i % 2].acorr(out[:,i] - np.mean(out[:,i]), maxlags=100)
    axis[i // 2, i % 2].set_title(f'ACF for variable {i}')
plt.savefig("jax-acf.png")
#plt.show()



# eof
