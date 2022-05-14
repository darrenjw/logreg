# logreg

## Bayesian inference for a logistic regression model in various languages and with various libraries

**This repo is a work-in-progress. Once it is reasonably complete, I'll write a post or two about it over on my [blog](https://darrenjw.wordpress.com/).**

This repos contains code for fully Bayesian inference for a logistic regression model using R, Python and Scala, using bespoke hand-coded samplers (both random walk Metropolis and MALA), and samplers constructed with the help of libraries such as JAGS, Stan, JAX, BlackJAX, and Spark.

### R

* [create-dataset.R](R/create-dataset.R) - we will use the infamous `MASS::Pima.tr` dataset, exported from R in [parquet](https://parquet.apache.org/) format (rather than CSV, as it's not 1993 any more).
* [fit-glm.R](R/fit-glm.R) - kick-off with a simple GLM fit in R for sanity-checking purposes.
* [fit-bayes.R](R/fit-bayes.R) - 

### Python

* 

### Scala

* 

