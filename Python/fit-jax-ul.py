#!/usr/bin/env python3
# fit-jax-ul.py
# Unadjusted Langevin using JAX (approximate)

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

# Use JAX auto-diff to compute gradient and Hessian

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

print("Next, unadjusted Langevin (approximate). Be patient...")

def ulKernel(lpi, glpi, dt = 1e-4, pre = 1):
    p = len(init)
    sdt = jnp.sqrt(dt)
    spre = jnp.sqrt(pre)
    advance = jit(lambda x: x + 0.5*pre*glpi(x)*dt)
    @jit
    def kernel(key, x):
        return advance(x) + jax.random.normal(key, [p])*spre*sdt
    return kernel

def mcmc(init, kernel, thin = 10, iters = 10000):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, iters)
    @jit
    def step(x, k):
        x = kernel(k, x)
        return x, x
    @jit
    def iter(x, k):
        keys = jax.random.split(k, thin)
        _, states = jax.lax.scan(step, x, keys)
        final = states[thin-1]
        return final, final
    x = init
    _, states = jax.lax.scan(iter, x, keys)
    return states

pre = jnp.array([100.,1.,1.,1.,1.,1.,25.,1.]).astype(jnp.float32)

out = mcmc(beta, ulKernel(lpost, glp, dt=1e-6, pre=pre), thin=4000)

print(out)
odf = pd.DataFrame(np.asarray(out), columns=["b0","b1","b2","b3","b4","b5","b6","b7"])
odf.to_parquet("fit-jax-ul.parquet")
print("Posterior summaries:")
summ = scipy.stats.describe(out)
print(summ)
print("\nMean: " + str(summ.mean))
print("Variance: " + str(summ.variance))



# eof
