#!/usr/bin/env python3
# fit-blackjax.py
# Fit using BlackJAX

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

import blackjax

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

print("Next, MH. Be patient...")
      
def mcmc(init, sampler, kernel, thin = 10, iters = 10000):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, iters)
    @jit
    def step(s, k):
        s, _ = kernel(k, s)
        return s, s
    @jit
    def fullIter(s, k):
        keys = jax.random.split(k, thin)
        _, states = jax.lax.scan(step, s, keys)
        final = sampler.init(states.position[thin-1,:])
        return final, final
    _, states = jax.lax.scan(fullIter, init, keys)
    return states[0]

pre = jnp.array([10.,1.,1.,1.,1.,1.,5.,1.]).astype(jnp.float32)

samp = blackjax.rmh(lpost, 0.02*pre)
bjInit = samp.init(beta)
out = mcmc(bjInit, samp, jit(samp.step), thin=1000)

print(out)
odf = pd.DataFrame(np.asarray(out), columns=["b0","b1","b2","b3","b4","b5","b6","b7"])
odf.to_parquet("fit-blackjax.parquet")
print("Posterior summaries:")
summ = scipy.stats.describe(out)
print(summ)
print("\nMean: " + str(summ.mean))
print("Variance: " + str(summ.variance))



# eof
