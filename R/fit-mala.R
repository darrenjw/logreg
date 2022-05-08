## fit-mala.R
## Bayesian estimation using gradient information (MALA)

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

lprior = function(beta)
    dnorm(beta[1], 0, 10, log=TRUE) + sum(dnorm(beta[-1], 0, 1, log=TRUE))

lpost = function(beta) ll(beta) + lprior(beta)

glp = function(beta) as.vector(t(X) %*% (y - 1/(1 + exp(-X %*% beta))))

print("MAP:")
fit = optim(init, lpost, method="BFGS", control=list(fnscale=-1, maxit=1000))
print(fit)
print(fit$par)

print("Next, MALA:")

mala = function(init, lpi, glpi, dt = 1e-4, pre = 1, ...) {
    p = length(init)
    sdt = sqrt(dt)
    spre = sqrt(pre)
    advance = function(x) x + 0.5*pre*glpi(x)*dt
    metropolisHastings(init, lpost,
                       function(x) rnorm(p, advance(x), spre*sdt),
                       function(new, old, log=TRUE) sum(dnorm(new, advance(old), spre*sdt, log)),
                       ...)
}

#out = mala(fit$par, lpost, glp, dt=1e-5, pre=c(100,1,1,1,1,1,25,1), thin=1000)
out = mala(init, lpost, glp, dt=1e-5, pre=c(100,1,1,1,1,1,25,1), thin=1000)

mcmcSummary(out)
image(cor(out)[ncol(out):1,])
pairs(out[sample(1:10000,1000),],pch=19,cex=0.2)




## eof
