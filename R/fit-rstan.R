#!/usr/bin/env Rscript
## fit-rstan.R
## Bayesian MCMC fit using R-Stan

if (!require("pacman")) install.packages("pacman")
pacman::p_load("arrow", "smfsb", "rstan")

df = read_parquet(file.path("..", "pima.parquet"))
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

modelstring="
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

"
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
constants = list(n=n, p=p, X=X, y=as.integer(y))
print("Calling stan now...")
thin = 2 # set thinning here
init_fun = function(...)
    list(beta=init) # same init for each chain not ideal...
output = stan(model_code=modelstring, data=constants, iter=2500*thin+1000,
              chains=4, warmup=1000, thin=thin, init=init_fun)
out = as.matrix(output)
mcmcSummary(out)
image(cor(out)[ncol(out):1,])
pairs(out[sample(1:10000,1000),],pch=19,cex=0.2)



## eof

