#!/usr/bin/env Rscript
## fit-bayes.R
## Bayesian estimation using log-likelihood

if (!require("pacman")) install.packages("pacman")
pacman::p_load("arrow", "smfsb")

df = read_parquet(file.path("..", "pima.parquet"))
print(head(df))
p = dim(df)[2]
y = df[, p]
y = as.integer(y)-1
X = as.matrix(df[, -p])
X = cbind(Int=1, X)
print(y[1:6])
print(head(X))

ll = function(beta)
    sum(-log(1 + exp(-(2*y - 1)*(X %*% beta))))

init = rnorm(p, 0.1)
names(init) = colnames(X)

print("MLE to start with:")
fit = optim(init, ll, method="BFGS", control=list(fnscale=-1, maxit=1000))
print(fit)
print(fit$par)

lprior = function(beta)
    dnorm(beta[1], 0, 10, log=TRUE) + sum(dnorm(beta[-1], 0, 1, log=TRUE))

lpost = function(beta) ll(beta) + lprior(beta)

print("MAP next:")
fit = optim(init, lpost, method="BFGS", control=list(fnscale=-1, maxit=1000))
print(fit)
print(fit$par)

print("Next, MH")

mhKernel = function(logPost, rprop, dprop = function(new, old, ...) { 1 })
    function(x, ll) {
        prop = rprop(x)
        llprop = logPost(prop)
        a = llprop - ll + dprop(x, prop) - dprop(prop, x)
        if (log(runif(1)) < a)
            list(x=prop, ll=llprop)
        else
            list(x=x, ll=ll)
    }
    
mcmc = function(init, kernel, iters = 10000, thin = 10, verb = TRUE) {
    p = length(init)
    ll = -Inf
    mat = matrix(0, nrow = iters, ncol = p)
    colnames(mat) = names(init)
    x = init
    if (verb) 
        message(paste(iters, "iterations"))
    for (i in 1:iters) {
        if (verb) 
            message(paste(i, ""), appendLF = FALSE)
        for (j in 1:thin) {
            pair = kernel(x, ll)
            x = pair$x
            ll = pair$ll
            }
        mat[i, ] = x
        }
    if (verb) 
        message("Done.")
    mat
}

out = mcmc(fit$par, mhKernel(lpost, function(x) c(rnorm(1, x[1], 0.2),
                                      rnorm(p-1, x[-1], 0.02))), thin=1000)
mcmcSummary(out)
image(cor(out)[ncol(out):1,])
pairs(out[sample(1:10000,1000),],pch=19,cex=0.2)

## eof

