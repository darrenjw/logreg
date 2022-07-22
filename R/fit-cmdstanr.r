#!/usr/bin/env Rscript
## fit-cmdstanr.R
## Bayesian MCMC fit using R-Stan

if (!require("pacman")) install.packages("pacman")
pacman::p_load("arrow", "smfsb", "cmdstanr")

df = read_parquet(file.path("pima.parquet"))
print(head(df))
n = dim(df)[1]
p = dim(df)[2]
y = df[, p]
y = as.integer(y)-1
X = as.matrix(df[, -p])
X = cbind(Int=1, X)
print(y[1:6])
print(head(X))

set.seed(43) # fix seed to avoid choosing bad inits
init = rnorm(p, 0.1)
names(init) = colnames(X)

modelstring ="
data {
  int<lower=1> n;
  int<lower=1> p;
  array[n] int<lower=0, upper=1> y;
  matrix[n, p] X;
}
parameters {
  vector[p] beta;
}
model {
 beta[1] ~ normal(0, 10);
 beta[2:p] ~ std_normal();
 y ~ bernoulli_logit_glm(X[:, 2:p], beta[1], beta[2:p]);
}
"

f <- write_stan_file(modelstring)
constants = list(n=n, p=p, X=X, y=as.integer(y))
thin = 2 # set thinning here

print("Calling stan now...")
mod <- cmdstan_model(f)

mod_out <- mod$sample(
  data = constants,
  parallel_chains = 4,
  iter_sampling = 2500 * thin + 1000,
  iter_warmup = 1000,
  thin = thin,
  init = function() list(beta = init)
)

out <- mod_out$summary()
out_draws <- mod_out$draws(format = "matrix")
image(cor(out_draws))
pairs(out_draws[sample(1:10000,1000),],pch=19,cex=0.2)
## eof

