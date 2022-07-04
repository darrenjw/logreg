# logreg

## Bayesian inference for a logistic regression model in various languages and with various libraries

**This repo is a work-in-progress. Once it is reasonably complete, I'll write a post or two about it over on my [blog](https://darrenjw.wordpress.com/).**

This repo contains code for MCMC-based fully Bayesian inference for a logistic regression model using [R](https://www.r-project.org/), [Python](https://www.python.org/), [Scala](https://www.scala-lang.org/), [Haskell](https://www.haskell.org/), [Dex](https://github.com/google-research/dex-lang), [Julia](https://julialang.org/) and [C](https://en.wikipedia.org/wiki/C_(programming_language)), using bespoke hand-coded samplers ([random walk Metropolis](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm), unadjusted Langevin algorithm, [MALA](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm), and [HMC](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo)), and samplers constructed with the help of libraries such as [JAGS](https://sourceforge.net/projects/mcmc-jags/), [Stan](https://mc-stan.org/), [JAX](https://jax.readthedocs.io/), [BlackJAX](https://blackjax-devs.github.io/blackjax/), [NumPyro](https://github.com/pyro-ppl/numpyro), [PyMC3](https://docs.pymc.io/en/v3/), and [Spark](https://spark.apache.org/). 

I intend to add similar examples using one or two other libraries. At some point I'd also like to switch to a much bigger dataset, that better illustrates some of the scalability issues of the different languages and libraries.

## The model

Here we will conduct fully Bayesian inference for the typical Bayesian [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) model for a binary outcome based on some covariates. The $i$th observation will be 1 with probability $p_i$, and the [logit](https://en.wikipedia.org/wiki/Logit) of $p_i$ will depend linearly on predictors. This leads to a log-likelihood function

$$l(b; y) = -\mathbb{1}'[\log(\mathbb{1} + \exp[-(2y - \mathbb{1})\circ(Xb)])]$$

where $y$ is a binary vector of responses, $X$ is an $n\times p$ matrix of covariates and $b$ is the $p$-vector of parameters of inferential interest.

JAX can auto-diff likelihoods like this, but for comparison purposes, we can also use hard-coded gradients for MALA and HMC:

$$\nabla l(b) = X'(y-p), \quad \text{where}\quad p = (\mathbb{1} + \exp[-Xb])^{-1}.$$

For a fully Bayesian analysis, we also need a prior distribution. Here we will assume independent normal priors on the elements of $b$. That is, $b_i \sim N(0, v_i)$. Note that the gradient of the log of this prior is

$$\nabla \pi(b) = -b\circ v^{-1}.$$

We will be analysing the "Pima" training dataset, with 200 observations and 7 predictors. Including an intercept as the first covariate gives a parameter vector of length $p=8$. The prior standard deviation for the intercept is 10, and for the other covariates is 1.


### R

Note that these scripts use [pacman](https://cran.r-project.org/web/packages/pacman/) to download and install any missing dependencies.

* [create-dataset.R](R/create-dataset.R) - we will use the infamous `MASS::Pima.tr` dataset, exported from R in [parquet](https://parquet.apache.org/) format (rather than CSV, as it's now the 21st Century, but also save in a simple text format for languages that can't easily read parquet...).
* [fit-glm.R](R/fit-glm.R) - kick-off with a simple GLM fit in R for sanity-checking purposes.
* [fit-bayes.R](R/fit-bayes.R) - MAP, followed by a Random walk Metropolis MCMC sampler in R.
* [fit-ul.R](R/fit-ul.R) - Unadjusted Langevin in R (with a simple diagonal pre-conditioner). Note that this algorithm is *approximate*, so we wouldn't expect it to match up perfectly with the exact sampling methods.
* [fit-mala.R](R/fit-mala.R) - MALA in R (with a diagonal pre-conditioner).
* [fit-hmc.R](R/fit-hmc.R) - HMC in R (with a diagonal mass-matrix).
* [fit-rjags.R](R/fit-rjags.R) - Fit using rjags. Note that this script probably won't work unless a site-wide installation of JAGS is available. 
* [fit-rstan.R](R/fit-rstan.R) - Fit using rstan.


### Python

These scripts assume a Python installation with [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/). The later scripts require [JAX](https://jax.readthedocs.io/). The BlackJAX scripts require [BlackJAX](https://blackjax-devs.github.io/blackjax/), the NumPyro script requires [NumPyro](https://github.com/pyro-ppl/numpyro), and the PyMC3 script requires [PyMC3](https://docs.pymc.io/en/v3/). These can be `pip` installed for basic use. See the websites for more detailed information.

* [fit-numpy.py](Python/fit-numpy.py) - Random walk MH with NumPy.
* [fit-np-ul.py](Python/fit-np-ul.py) - Unadjusted Langevin with NumPy (approximate).
* [fit-np-mala.py](Python/fit-np-mala.py) - MALA with NumPy.
* [fit-np-hmc.py](Python/fit-np-hmc.py) - HMC with NumPy.
* [fit-jax.py](Python/fit-jax.py) - RM MH with log posterior and MH kernel in JAX, but main MCMC loop in python.
* [fit-jax2.py](Python/fit-jax2.py) - As above, but with main MCMC loop in JAX (much faster).
* [fit-jax-ul.py](Python/fit-jax-ul.py) - JAX for unadjusted Langevin (with a diagonal pre-conditioner). Note that this is an approximate algorithm. Note also that JAX AD is being used for gradients.
* [fit-jax-mala.py](Python/fit-jax-mala.py) - JAX for MALA (with a diagonal pre-conditioner). JAX AD for gradients.
* [fit-jax-hmc.py](Python/fit-jax-hmc.py) - JAX for HMC (with a diagonal mass-matrix). JAX AD for gradients.
* [fit-blackjax.py](Python/fit-blackjax.py) - RW MH using BlackJAX.
* [fit-blackjax-mala.py](Python/fit-blackjax-mala.py) - MALA with BlackJAX. Note that the MALA kernel in BlackJAX doesn't seem to allow a pre-conditioner, so a huge thinning interval is used here to get vaguely reasonable results.
* [fit-blackjax-nuts.py](Python/fit-blackjax-nuts.py) - NUTS sampler from BlackJAX.
* [fit-numpyro.py](Python/fit-numpyro.py) - NUTS sampler from NumPyro.
* [fit-pymc3.py](Python/fit-pymc3.py) - NUTS sampler from PyMC3.


### Scala

The Scala examples just require a recent JVM and [sbt](https://www.scala-sbt.org/). `sbt` will look after other dependencies (including Scala itself). See the [Readme](Scala/Readme.md) in the Scala directory for further info.

* [fit-bayes.scala](Scala/lr/src/main/scala/fit-bayes.scala) - Random walk MH with Scala and Breeze.
* [fit-nopar.scala](Scala/lr/src/main/scala/fit-nopar.scala) - Random walk MH, re-factored to make it easy to run in parallel, but still serial.
* [fit-par.scala](Scala/lr/src/main/scala/fit-par.scala) - Random walk MH, running in parallel on all available CPU cores. Note that the evaluation of the log-likelihood is parallelised over observations, but due to the very small size of this dataset, this version runs considerably slower than the previous version. For large datasets it will be a different story.
* [fit-ul.scala](Scala/lr/src/main/scala/fit-ul.scala) - Unadjusted Langevin with Breeze (approximate).
* [fit-mala.scala](Scala/lr/src/main/scala/fit-mala.scala) - MALA with Breeze.
* [fit-hmc.scala](Scala/lr/src/main/scala/fit-hmc.scala) - HMC with Breeze.

#### Scala with Spark

The Spark example requires a Spark installation in addition to `sbt`. See the [Readme](Scala/Readme.md) in the Scala directory for further info.

* [fit-spark.scala](Scala/spark/src/main/scala/fit-spark.scala) - RW MH, with Spark being used to distribute the log-likelihood evaluation over a cluster. Note that this code runs very slowly, as the overheads associated with distributing the computation dominate for very small datasets like the one used here. The thinning interval has been reduced so that the job completes in reasonable time.


### Haskell

The Haskell examples use [stack](https://docs.haskellstack.org/en/stable/README/) to build and run and manage dependencies. See the [readme](Haskell/lr/README.md) in the `Haskell/lr` directory for further details.

* [Rwmh.hs](Haskell/lr/app/Rwmh.hs) - Random walk MH in Haskell, using a stateful monadic random number generator.
* [RwmhP.hs](Haskell/lr/app/RwmhP.hs) - Random walk MH in Haskell, using a pure random number generator explicity threaded through the code.
* [RwmhPS.hs](Haskell/lr/app/RwmhPS.hs) - Random walk MH in Haskell, using a pure random number generator together with a splitting approach, *a la* JAX and Dex.
* [Mala.hs](Haskell/lr/app/Mala.hs) - MALA in Haskell (using a stateful monadic random number generator).
* [Hmc.hs](Haskell/lr/app/Hmc.hs) - HMC in Haskell (using a stateful monadic random number generator).


### Dex

The [Dex](https://github.com/google-research/dex-lang) examples rely only on a basic Dex installation. Note that Dex is an early-stage research project lacking many of the tools and libraries one would normally expect. It's also rather lacking documentation. However, it's interesting, purely functional, strongly typed, and fast.

* [fit-bayes.dx](Dex/fit-bayes.dx) - Random walk MH in Dex. Dex uses a splittable random number generator, similar to JAX. It's not quite as fast as JAX, but faster than anything else I've tried, including my C code.
* [fit-mala.dx](Dex/fit-mala.dx) - MALA in Dex, with hard-coded gradients.
* [fit-mala-ad.dx](Dex/fit-mala-ad.dx) - MALA in Dex, with hard-coded gradients. Roughly half as fast as using hard-coded gradients, which seems reasonable.


### Julia

The Julia examples depend only on standard packages which are part of the Julia package ecosystem, and are therefore easy to `add`, in principle. But in my limited experience, package dependency conflicts are even more of a problem in Julia than they are in Python, and that's saying something.

* [fit-bayes.jl](Julia/fit-bayes.jl) - Random walk MH in Julia.
* [fit-mala.jl](Julia/fit-mala.jl) - MALA in Julia, with hard-coded gradients.
* [fit-mala-ad.jl](Julia/fit-mala-ad.jl) - MALA in Julia, with AD gradients via Zygote. Note that this is much slower than using hard-coded gradients.
* [fit-hmc.jl](Julia/fit-hmc.jl) - HMC in Julia.
* [fit-hmc-ad.jl](Julia/fit-hmc-ad.jl) - HMC in Julia, using AD for gradients. Again, this is much slower than using hard-coded gradients.


### C

The C examples assume a Unix-like development environment. See the [Readme](C/Readme.md) in the C directory for further info.

* [fit-bayes.c](C/fit-bayes.c) - Random walk MH with C and the GSL. The code isn't pretty, but it's fast (in particular, there are no allocations in the main MCMC loop). But still not as fast as JAX, even on a single core.





**Copyright (C) 2022, Darren J Wilkinson**, but released under a GPL-3.0 license
