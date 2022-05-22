#!/usr/bin/env python3
# fit-numpyro.py
# Fit using NumPyro

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

import numpyro
import numpyro.distributions as dist
from numpyro.infer import HMC, MCMC, NUTS

df = pd.read_parquet(os.path.join("..", "pima.parquet"))
print(df)
n, p = df.shape
print(n, p)

y = pd.get_dummies(df["type"])["Yes"].to_numpy(dtype='float32')
X = df.drop(columns="type").to_numpy()
X = np.hstack((np.ones((n,1)), X))
print(X)
print(y)
X = X.astype(jnp.float32)
y = y.astype(jnp.float32)

# Now specify model in NumPyro
pscale = jnp.array([10.,1.,1.,1.,1.,1.,1.,1.]).astype(jnp.float32)
def model(X, y):
    coefs = numpyro.sample("beta", dist.Normal(jnp.zeros(p), pscale))
    logits = jnp.dot(X, coefs)
    return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

rng_key = jax.random.PRNGKey(42)
step_size = jnp.sqrt(0.1 / p)
trajectory_length = step_size * 20
kernel = NUTS(model)
thin = 1
mcmc = MCMC(kernel, num_warmup=1000, thinning=thin, num_samples=10000*thin)
print("Running MCMC now...")
mcmc.run(rng_key, X, y)
print("MCMC finished.")
mcmc.print_summary()
out = mcmc.get_samples()['beta']

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
plt.savefig("numpyro-nuts-trace.png")
#plt.show()

figure, axis = plt.subplots(4, 2)
for i in range(8):
    axis[i // 2, i % 2].hist(out[:,i], 50)
    axis[i // 2, i % 2].set_title(f'Histogram for variable {i}')
plt.savefig("numpyro-nuts-hist.png")
#plt.show()

figure, axis = plt.subplots(4, 2)
for i in range(8):
    axis[i // 2, i % 2].acorr(out[:,i] - np.mean(out[:,i]), maxlags=100)
    axis[i // 2, i % 2].set_title(f'ACF for variable {i}')
plt.savefig("numpyro-nuts-acf.png")
#plt.show()



# eof
