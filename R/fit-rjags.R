## fit-rjags.R
## Bayesian MCMC fit using R-JAGS

if (!require("pacman")) install.packages("pacman")
pacman::p_load("arrow", "smfsb", "rjags")

df = read_parquet(file.path("..", "pima.parquet"))
print(head(df))
p = dim(df)[2]
y = df[, p]
y = as.integer(y)-1
X = as.matrix(df[, -p])
X = cbind(Int=1, X)
print(y[1:6])
print(head(X))






## eof

