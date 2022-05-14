# logreg

## Bayesian inference for a logistic regression model in various languages and with various libraries

**This repo is a work-in-progress. Once it is reasonably complete, I'll write a post or two about it over on my [blog](https://darrenjw.wordpress.com/).**

This repo contains code for MCMC-based fully Bayesian inference for a logistic regression model using [R](https://www.r-project.org/), [Python](https://www.python.org/) and [Scala](https://www.scala-lang.org/), using bespoke hand-coded samplers (both random walk Metropolis and MALA), and samplers constructed with the help of libraries such as [JAGS](https://sourceforge.net/projects/mcmc-jags/), [Stan](https://mc-stan.org/), [JAX](https://jax.readthedocs.io/), [BlackJAX](https://blackjax-devs.github.io/blackjax/), and [Spark](https://spark.apache.org/). I intend to very soon add similar examples in [DEX](https://github.com/google-research/dex-lang) and [C](https://en.wikipedia.org/wiki/C_(programming_language)).

## The model

**Model and prior here, including likelihood, prior, gradient and hessian...**

### R

Note that these scripts use [pacman](https://cran.r-project.org/web/packages/pacman/) to download and install any missing dependencies.

* [create-dataset.R](R/create-dataset.R) - we will use the infamous `MASS::Pima.tr` dataset, exported from R in [parquet](https://parquet.apache.org/) format (rather than CSV, as it's not 1993 any more).
* [fit-glm.R](R/fit-glm.R) - kick-off with a simple GLM fit in R for sanity-checking purposes.
* [fit-bayes.R](R/fit-bayes.R) - MAP, followed by a Random walk Metropolis MCMC sampler in R.
* [fit-mala.R](R/fit-mala.R) - MALA in R (with a simple diagonal pre-conditioner).
* [fit-rjags.R](R/fit-rjags.R) - Fit using rjags. Note that this script probably won't work unless a site-wide installation of JAGS is available. 
* [fit-stan.R](R/fit-rstan.R) - Fit using rstan.

### Python

These scripts assume a Python installation with [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/). The later scripts require [JAX](https://jax.readthedocs.io/) and [BlackJAX](https://blackjax-devs.github.io/blackjax/), which can be `pip` installed for basic use. See the websites for more detailed information.

* [fit-numpy.py](Python/fit-numpy.py) - Random walk MH with NumPy.
* [fit-np-mala.py](Python/fit-np-mala.py) - MALA with NumPy.
* [fit-jax.py](Python/fit-jax.py) - RM MH with log posterior and MH kernel in JAX, but main MCMC loop in python.
* [fit-jax2.py](Python/fit-jax2.py) - As above, but with main MCMC loop in JAX (much faster).
* [fit-jax-mala.py](Python/fit-jax-mala.py) - JAX for MALA (with a simple diagonal pre-conditioner).
* [fit-blackjax.py](Python/fit-blackjax.py) - RW MH using BlackJAX.
* [fit-blackjax-mala.py](Python/fit-blackjax-mala.py) - MALA with BlackJAX. Note that the MALA kernel in BlackJAX doesn't seem to allow a pre-conditioner, so a huge thinning interval is used here to get vaguely reasonable results.
* [fit-blackjax-nuts.py](Python/fit-blackjax-nuts.py) - NUTS sampler from BlackJAX.

### Scala

* 

