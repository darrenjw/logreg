#!/usr/bin/env julia
# fit-bayes.jl
# Fit Bayesian logistic regression using RW MH MCMC in Julia

# Should I use Julia? Maybe, maybe not...  https://yuri.is/not-julia/


using ParquetFiles, DataFrames, Random, Distributions, Plots, StatsBase

# Define some functions

function lprior(beta)
    d = Normal(0,1)
    logpdf(Normal(0,10), beta[1]) + sum(map((bi) -> logpdf(d, bi), beta[2:8]))
end

ll(beta) = sum(-log.(1.0 .+ exp.(-(2*y .- 1.0).*(x * beta))))

lpost(beta) = lprior(beta) + ll(beta)

function mhKernel(logPost, rprop)
    function kern(x, ll)
        prop = rprop(x)
        llprop = logPost(prop)
        a = llprop - ll
        if (log(rand(rng)) < a)
            return (prop, llprop)
        else
            return (x, ll)
        end
    end
    kern
end

function mcmc(init, kernel, iters, thin)
    p = length(init)
    ll = -Inf
    mat = zeros(iters, p)
    x = init
    for i in 1:iters
        print(i); print(" ")
        for j in 1:thin
            x, ll = kernel(x, ll)
        end
        mat[i,:] = x
    end
    println(".")
    mat
end


# Main execution thread

# Load and process the data
df = DataFrame(load("../pima.parquet"))
y = df.type
y = map((yi) -> yi == "Yes" ? 1.0 : 0.0, y)
x = df[:,1:7]
x = Matrix(x)
x = hcat(ones(200, 1), x)
# Set up for doing MCMC
beta = zeros(8, 1)
beta[1] = -10
rng = MersenneTwister(1234)
norm = Normal(0, 0.02)
kern = mhKernel(lpost, x ->
                vcat( x[1]+rand(rng, norm)*10, x[2:8]+rand!(rng, norm, zeros(7))))
# Main MCMC loop
out = mcmc(beta, kern, 10000, 1000)

# Plot results
plot(1:10000, out, layout=(4, 2))
savefig("trace.pdf")
histogram(out, layout=(4, 2))
savefig("hist.pdf")
plot(1:400, autocor(out, 1:400), layout = (4, 2))
savefig("acf.pdf")


# eof
