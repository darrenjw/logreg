#!/usr/bin/env julia
# fit-mala-ad.jl
# Fit Bayesian logistic regression using MALA MCMC in Julia

using ParquetFiles, DataFrames, Random, Distributions, Plots, StatsBase, Zygote

# Define some functions

function lprior(beta)
    d = Normal(0,1)
    logpdf(Normal(0,10), beta[1]) + sum(map((bi) -> logpdf(d, bi), beta[2:8]))
end

ll(beta) = sum(-log.(1.0 .+ exp.(-(2*y .- 1.0).*(x * beta))))

lpost(beta) = lprior(beta) + ll(beta)

function malaKernel(lpi, dt, pre)
    sdt = sqrt(dt)
    spre = sqrt.(pre)
    glpi = lpi' # use Zygote to AD the gradient...
    norm = Normal(0,1)
    advance(x) = x .+ ( (0.5 * dt) .* (pre .* glpi(x)) )
    function dprop(n, o)
        ao = advance(o)
        sum( map((i) -> logpdf(Normal(ao[i], spre[i]*sdt), n[i]), 1:8) )
    end
    mhKernel(lpi, x -> advance(x) .+ rand!(rng, norm, zeros(8)).*spre.*sdt,
             dprop)
end

function mhKernel(logPost, rprop, dprop)
    function kern(x, ll)
        prop = rprop(x)
        llprop = logPost(prop)
        a = llprop - ll + dprop(x, prop) - dprop(prop, x)
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
kern = malaKernel(lpost, 1e-5, [100.0, 1, 1, 1, 1, 1, 25, 1])
                  
# Main MCMC loop
out = mcmc(beta, kern, 10000, 1000)

# Plot results
plot(1:10000, out, layout=(4, 2))
savefig("trace-mala-ad.pdf")
histogram(out, layout=(4, 2))
savefig("hist-mala-ad.pdf")
plot(1:400, autocor(out, 1:400), layout = (4, 2))
savefig("acf-mala-ad.pdf")

# eof

