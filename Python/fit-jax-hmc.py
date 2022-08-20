#!/usr/bin/env python3
# fit-jax-hmc.py
# HMC using JAX

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

print("Next, HMC. Be patient...")

def mhKernel(lpost, rprop, dprop = jit(lambda new, old: 1.)):
    @jit
    def kernel(key, x):
        key0, key1 = jax.random.split(key)
        prop = rprop(key0, x)
        ll = lpost(x)
        lp = lpost(prop)
        a = lp - ll + dprop(x, prop) - dprop(prop, x)
        accept = (jnp.log(jax.random.uniform(key1)) < a)
        return jnp.where(accept, prop, x)
    return kernel

def hmcKernel(lpi, glpi, eps = 1e-4, l = 10, dmm = 1):
    sdmm = jnp.sqrt(dmm)
    @jit
    def leapf(q, p):    
        p = p + 0.5*eps*glpi(q)
        for i in range(l): # TODO: Replace with a jax.lax.fori_loop ??
            q = q + eps*p/dmm
            if (i < l-1):
                p = p + eps*glpi(q)
            else:
                p = p + 0.5*eps*glpi(q)
        return jnp.concatenate((q, -p))
    @jit
    def alpi(x):
        d = len(x) // 2
        return lpi(x[jnp.array(range(d))]) - 0.5*jnp.sum((x[jnp.array(range(d,2*d))]**2)/dmm)
    @jit
    def rprop(k, x):
        d = len(x) // 2
        return leapf(x[jnp.array(range(d))], x[jnp.array(range(d, 2*d))])
    mhk = mhKernel(alpi, rprop)
    @jit
    def kern(k, q):
        key0, key1 = jax.random.split(k)
        d = len(q)
        x = jnp.concatenate((q, jax.random.normal(key0, [d])*sdmm))
        return mhk(key1, x)[jnp.array(range(d))]
    return kern
    
        
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

out = mcmc(beta, hmcKernel(lpost, glp, eps=1e-3, l=50, dmm=1/pre), thin=20)

print(out)
odf = pd.DataFrame(out, columns=["b0","b1","b2","b3","b4","b5","b6","b7"])
odf.to_parquet("fit-jax-hmc.parquet")
print("Posterior summaries:")
summ = scipy.stats.describe(out)
print(summ)
print("\nMean: " + str(summ.mean))
print("Variance: " + str(summ.variance))



# eof
