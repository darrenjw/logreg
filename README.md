# logreg

## Bayesian inference for a logistic regression model in various languages and with various libraries

**This repo is a work-in-progress. Once it is reasonably complete, I'll write a post or two about it over on my [blog](https://darrenjw.wordpress.com/).**

This repo contains code for fully Bayesian inference for a logistic regression model using [R](https://www.r-project.org/), [Python](https://www.python.org/) and [Scala](https://www.scala-lang.org/), using bespoke hand-coded samplers (both random walk Metropolis and MALA), and samplers constructed with the help of libraries such as [JAGS](https://sourceforge.net/projects/mcmc-jags/), [Stan](https://mc-stan.org/), [JAX](https://jax.readthedocs.io/), [BlackJAX](https://blackjax-devs.github.io/blackjax/), and [Spark](https://spark.apache.org/). I intend to very soon add similar examples in [DEX](https://github.com/google-research/dex-lang) and [C](https://en.wikipedia.org/wiki/C_(programming_language)).

## The model

**Model and prior here, including likelihood, prior, gradient and hessian...**

### R

Note that these scripts use [pacman](https://cran.r-project.org/web/packages/pacman/) to download and install any missing dependencies.

* [create-dataset.R](R/create-dataset.R) - we will use the infamous `MASS::Pima.tr` dataset, exported from R in [parquet](https://parquet.apache.org/) format (rather than CSV, as it's not 1993 any more).
* [fit-glm.R](R/fit-glm.R) - kick-off with a simple GLM fit in R for sanity-checking purposes.
* [fit-bayes.R](R/fit-bayes.R) - Random walk Metropolis MCMC sampler in R.
* [fit-mala.R](R/fit-mala.R) - MALA in R (with a simple diagonal pre-conditioner).
* [fit-rjags.R](R/fit-rjags.R) - Fit using rjags. Note that this script probably won't work unless a site-wide installation of JAGS is available. 
* [fit-stan.R](R/fit-rstan.R) - Fit using rstan.

### Python

* [fit-numpy.py](Python/fit-numpy.py) - 
* 

### Scala

* 

