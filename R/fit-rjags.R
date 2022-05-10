## fit-rjags.R
## Bayesian MCMC fit using R-JAGS

if (!require("pacman")) install.packages("pacman")
pacman::p_load("arrow", "smfsb", "rjags")

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

set.seed(43)
init = rnorm(p, 0.1)
names(init) = colnames(X)
data = list(X=X, y=as.integer(y), p=p, n=n)
modelstring="
  model {
    for (i in 1:n) {
      y[i] ~ dbern(pr[i])
      logit(pr[i]) <- inprod(X[i,], beta)
    }
    beta[1] ~ dnorm(0, 0.1)
    for (j in 2:p) {
      beta[j] ~ dnorm(0, 1)
    }
  }
"
model=jags.model(textConnection(modelstring), data=data, inits=list(beta=init))
update(model,n.iter=100)
output=coda.samples(model=model,variable.names=c("beta"), n.iter=10000, thin=1)
print(summary(output))
plot(output)





## eof

