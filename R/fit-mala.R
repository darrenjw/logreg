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
print("without gradients")
fit = optim(init, lpost, method="BFGS", control=list(fnscale=-1, maxit=1000))
#print(fit)
print(fit$par)
print(ll(fit$par))
print(glp(fit$par))

print("with gradients")
fit = optim(init, lpost, glp, method="BFGS", control=list(fnscale=-1, maxit=1000))
#print(fit)
print(fit$par)
print(ll(fit$par))
print(glp(fit$par))

print("Next, MALA:")

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

malaKernel = function(lpi, glpi, dt = 1e-4, pre = 1) {
    sdt = sqrt(dt)
    spre = sqrt(pre)
    advance = function(x) x + 0.5*pre*glpi(x)*dt
    mhKernel(lpi, function(x) rnorm(p, advance(x), spre*sdt),
             function(new, old) sum(dnorm(new, advance(old), spre*sdt, log=TRUE)))
}


out = mcmc(fit$par, malaKernel(lpost, glp, dt=1e-5, pre=c(100,1,1,1,1,1,25,1)), thin=1000)

mcmcSummary(out)
image(cor(out)[ncol(out):1,])
pairs(out[sample(1:10000,1000),],pch=19,cex=0.2)




## eof
