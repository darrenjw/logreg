# Bayesian inference for a logistic regression model (Part 1)

## Part 1: The basics

### Introduction

This is the first in a series of posts on MCMC-based fully Bayesian inference for a logistic regression model. In this series we will look at the model, and see how the posterior distribution can be sampled using a variety of different programming languages and libraries. 

## Logistic regression

[Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) is concerned with predicting a binary outcome based on some covariate information. The probability of "success" is modelled via a logistic transformation of a linear predictor constructed from the covariate vector.

This is a very simple model, but is a convenient toy example since it is arguably the simplest interesting example of an intractable (nonlinear) statistical model requiring some kind of iterative numerical fitting method, even in the non-Bayesian setting. In a Bayesian context, the posterior distribution is intractable, necessitating either approximate or computationally intensive numerical methods of "solution". In this series of posts, we will mainly concentrate on MCMC algortithms for sampling the full posterior distribution of the model parameters given some observed data.

We assume $n$ observations and $p$ covariates (including an intercept term that is always 1). The binary observations $y_i,\ i=1,\ldots,n$ are 1 for a "success" and 0 for a "failure". The covariate $p$-vectors $x_i\, i=1,\ldots,n$ all have 1 as the first element. The statistical model is
$$
\operatorname{logit}\left(\operatorname{Pr}[Y_i = 1]\right) = x_i \cdot \beta,\quad i=1,\ldots,n,
$$
where $\beta$ is a $p$-vector of parameters, $a\cdot b = a^\textsf{T} b$, and
$$
\operatorname{logit}(q) \equiv \log\left(\frac{q}{1-q}\right),\quad \forall\, q\in (0,1).
$$
Equivalently,
$$
\operatorname{Pr}[Y_i = 1] = \operatorname{expit}(x_i \cdot \beta),\quad i=1,\ldots,n,
$$
where
$$
\operatorname{expit}(\theta) \equiv \frac{1}{1+e^{-\theta}},\quad \forall\, \theta\in\mathbb{R}.
$$
Note that the *expit* function is sometimes called the *logistic* or *sigmoid* function, but expit is slightly less ambiguous. The statistical problem is to choose the parameter vector $\beta$ to provide the "best" model for the probability of success. In the Bayesian setting, a prior distribution (typically multivariate normal) is specified for $\beta$, and then the posterior distribution after conditioning on the data is the object of inferential interest.

### Example problem

In order to illustrate the ideas, it is useful to have a small running example. Here we will use the (infamous) Pima training dataset (`MASS::Pima.tr` in R). Here there are $n=200$ observations and 7 predictors. Adding an intercept gives $p=8$ covariates. For the Bayesian analysis, we need a prior on $\beta$. We will assume independent mean zero normal distributions for each component. The prior standard deviation for the intercept will be 10 and for the other covariates will be 1.

### Describing the model in some PPLs

In this first post in the series, we will use [probabilistic programming](https://en.wikipedia.org/wiki/Probabilistic_programming) languages (PPLs) to describe the model and sample the posterior distribution.

#### JAGS

[JAGS](https://sourceforge.net/projects/mcmc-jags/) is a stand-alone [domain specific language](https://en.wikipedia.org/wiki/Domain-specific_language) (DSL) for probabilistic programming. It can be used independently of general purpose programming languages, or called from popular languages for data science such as Python and R. We can describe our model in JAGS with the following code.

```R
  model {
    for (i in 1:n) {
      y[i] ~ dbern(pr[i])
      logit(pr[i]) <- inprod(X[i,], beta)
    }
    beta[1] ~ dnorm(0, 0.01)
    for (j in 2:p) {
      beta[j] ~ dnorm(0, 1)
    }
  }
```
Note that JAGS uses *precision* as the second parameter of a normal distribution. See the [full runnable R script](https://github.com/darrenjw/logreg/blob/main/R/fit-rjags.R) for further details. Given this model description, JAGS can constructe an MCMC sampler for the posterior distribution of the model parameters given the data. See the full script for how to feed in the data, run the sampler, and analyse the output.

#### Stan

[Stan](https://mc-stan.org/) is another stand-alone DSL for probabilistic programming, and has a very sophisticated sampling algorithm, making it a popular choice for non-trivial models. It uses gradient information for sampling, and therefore requires a differentiable log-posterior. We could encode our logistic regression model as follows.
```R
data {
  int<lower=1> n;
  int<lower=1> p;
  int<lower=0, upper=1> y[n];
  real X[n,p];
}
parameters {
  real beta[p];
}
model {
  for (i in 1:n) {
    real eta = dot_product(beta, X[i,]);
    real pr = 1/(1+exp(-eta));
    y[i] ~ binomial(1, pr);
  }
  beta[1] ~ normal(0, 10);
  for (j in 2:p) {
    beta[j] ~ normal(0, 1);
  }
}
```
Note that Stan uses *standard deviation* as the second parameter of the normal distribution. See the [full runnable R script](https://github.com/darrenjw/logreg/blob/main/R/fit-rstan.R) for further details.

#### PyMC

JAGS and Stan are *stand-alone* DSLs for probabilistic programming. This has the advantage of making them independent of any particular host (general purpose) programming language. But it also means that they are not able to take advantage of the language and tool support of an existing programming language. An alternative to stand-alone DSLs are *embedded* DSLs (eDSLs). Here, a DSL is embedded as a library or package within an existing (general purpose) programming language. Then, ideally, in the context of PPLs, probabilistic programs can become ordinary values within the host language, and this can have some advantages, especially if the host language is sophisticated. A number of probabilistic programming languages have been implemented as eDSLs in Python. [Python](https://www.python.org/) is not a particularly sophisticated language, so the advantages here are limited, but not negligible.

[PyMC](https://www.pymc.io/) is probably the most popular eDSL PPL in Python. We can encode our model in PyMC as follows.
```python
pscale = np.array([10.,1.,1.,1.,1.,1.,1.,1.])
with pm.Model() as model:
    beta = pm.Normal('beta', 0, pscale, shape=p)
    eta = pmm.matrix_dot(X, beta)
    pm.Bernoulli('y', logit_p=eta, observed=y)
    traces = pm.sample(2500, tune=1000, init="adapt_diag", return_inferencedata=True)
```
See the [full runnable script](https://github.com/darrenjw/logreg/blob/main/Python/fit-pymc.py) for further details.

#### NumPyro

[NumPyro](https://github.com/pyro-ppl/numpyro) is a fork of [Pyro](https://github.com/pyro-ppl/pyro) for NumPy and [JAX](https://jax.readthedocs.io/) (of which more later). We can encode our model with NumPyro as follows.
```python
pscale = jnp.array([10.,1.,1.,1.,1.,1.,1.,1.]).astype(jnp.float32)
def model(X, y):
    coefs = numpyro.sample("beta", dist.Normal(jnp.zeros(p), pscale))
    logits = jnp.dot(X, coefs)
    return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)
```
See the [full runnable script](https://github.com/darrenjw/logreg/blob/main/Python/fit-numpyro.py) for further details.

**Please note that none of the above examples have been optimised**, or are even necessarily expressed idiomatically within each PPL. I've just tried to express the model in as simple and similar way across each PPL. For example, I know that there is a function `bernoulli_logit_glm` in Stan which would simplify the model and improve sampling efficiency, but I've deliberately not used it in order to try and keep the implementations as basic as possible. The same will be true for all of the code examples in this series of blog posts. The code has not been optimised and should therefore not be used for serious benchmarking.

### Next steps

PPLs are convenient, and are becoming increasingly sophisticated. Each of the above PPLs provides a simple way to pass in observed data, and automatically construct an MCMC algorithm for sampling from the implied posterior distribution - see the full scripts for details. All of the PPLs work well for this problem, and all produce identical results up to Monte Carlo error. Each PPL has its own approach to sampler construction, and some PPLs offer multiple choices. However, more challenging problems often require highly customised samplers. Such samplers will often need to be created from scratch, and will require (at least) the ability to compute the (unnormalised log) posterior density at proposed parameter values, so in the next post we will look at how this can be derived for this model (in a couple of different ways) and coded up from scratch in a variety of programming languages.

All of the complete, runnable code associated with this series of blog posts can be obtained from [this public github repo](https://github.com/darrenjw/logreg).

