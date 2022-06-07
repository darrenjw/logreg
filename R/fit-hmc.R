#!/usr/bin/env Rscript
## fit-hmc.R
## Bayesian estimation using gradient information (HMC)

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

pscale = c(10, rep(1,7))

lprior = function(beta)
    sum(dnorm(beta, 0, pscale, log=TRUE))

lpost = function(beta) ll(beta) + lprior(beta)

glp = function(beta) {
    glpr = -beta/(pscale*pscale)
    gll = as.vector(t(X) %*% (y - 1/(1 + exp(-X %*% beta))))
    glpr + gll
}

print(init)
print(ll(init))
print(glp(init))

print("MAP:")
fit = optim(init, lpost, glp, method="BFGS", control=list(fnscale=-1, maxit=1000))
#print(fit)
print(fit$par)
print(ll(fit$par))
print(glp(fit$par))

print("Next, HMC:")

mhKernel = function(logPost, rprop)
    function(x) {
        prop = rprop(x)
        a = logPost(prop) - logPost(x)
        if (log(runif(1)) < a)
            prop
        else
            x
    }
    
mcmc = function(init, kernel, iters = 10000, thin = 10, verb = TRUE) {
    p = length(init)
    mat = matrix(0, nrow = iters, ncol = p)
    colnames(mat) = names(init[1:p])
    x = init
    if (verb) 
        message(paste(iters, "iterations"))
    for (i in 1:iters) {
        if (verb) 
            message(paste(i, ""), appendLF = FALSE)
        for (j in 1:thin)
            x = kernel(x)
        mat[i, ] = x
        }
    if (verb) 
        message("Done.")
    mat
}

hmcKernel = function(lpi, glpi, eps = 1e-4, l=10, dmm = 1) {
    sdmm = sqrt(dmm)
    leapf = function(q, p) {
        p = p + 0.5*eps*glpi(q)
        for (i in 1:l) {
            q = q + eps*p/dmm
            if (i < l)
                p = p + eps*glpi(q)
            else
                p = p + 0.5*eps*glpi(q)
        }
        c(q, -p)
    }
    alpi = function(x) {
        d = length(x)/2
        lpi(x[1:d]) - 0.5*sum((x[(d+1):(2*d)]^2)/dmm)
    }
    rprop = function(x) {
        d = length(x)/2
        leapf(x[1:d], x[(d+1):(2*d)])
    }
    mhk = mhKernel(alpi, rprop)
    function(q) {
        d = length(q)
        x = c(q, rnorm(d, 0, sdmm))
        mhk(x)[1:d]
    }
}

out = mcmc(fit$par,
           hmcKernel(lpost, glp, eps=1e-3, l=50, dmm=1/c(100,1,1,1,1,1,25,1)),
           thin=20)

mcmcSummary(out)
image(cor(out)[ncol(out):1,])
pairs(out[sample(1:10000,1000),],pch=19,cex=0.2)




## eof
