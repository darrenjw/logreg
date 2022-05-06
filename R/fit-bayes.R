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
out = metropolisHastings(fit$par, ll,
                         function(x) c(rnorm(1, x[1], 0.2),
                                       rnorm(p-1, x[-1], 0.02)),
                         dprior = function(x, log=TRUE) lprior(x),
                         thin=500)
mcmcSummary(out)
pairs(out)

## eof

