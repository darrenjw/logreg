#!/usr/bin/env julia
# fit-hmc-ad.jl
# Fit Bayesian logistic regression using HMC in Julia

using ParquetFiles, DataFrames, Random, Distributions, Plots, StatsBase, Zygote

# Define some functions

function lprior(beta)
    d = Normal(0,1)
    logpdf(Normal(0,10), beta[1]) + sum(map((bi) -> logpdf(d, bi), beta[2:8]))
end

ll(beta) = sum(-log.(1.0 .+ exp.(-(2*y .- 1.0).*(x * beta))))

lpost(beta) = lprior(beta) + ll(beta)

pscale = [10.0, 1, 1, 1, 1, 1, 1, 1]

function hmcKernel(lpi, eps, l, dmm)
    d = length(dmm)
    glpi = lpi'
    sdmm = sqrt.(dmm)
    norm = Normal(0,1)
    function leapf(q, p)
        p = p .+ (0.5*eps).*glpi(q)
        for i in 1:l
            q = q .+ eps.*(p./dmm)
            if (i < l)
                p = p .+ eps.*glpi(q)
            else
                p = p .+ (0.5*eps).*glpi(q)
            end
        end
        (q, -p)
    end
    function alpi(x)
        (q, p) = x
        lpi(q) - 0.5*sum((p.^2)./dmm)
    end
    function rprop(x)
        (q, p) = x
        leapf(q, p)
    end
    mhk = mhKernel(alpi, rprop)
    function (q0)
        p0 = rand!(rng, norm, zeros(d)).*sdmm
        (q, p) = mhk((q0, p0))
        q
    end
end

function mhKernel(logPost, rprop)
    function kern(x)
        prop = rprop(x)
        a = logPost(prop) - logPost(x)
        if (log(rand(rng)) < a)
            return prop
        else
            return x
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
            x = kernel(x)
        end
        mat[i,:] = x
    end
    println(".")
    mat
end

# Main execution thread

# Load and process the data
df = DataFrame(load(joinpath(@__DIR__, "..", "pima.parquet")))
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
kern = hmcKernel(lpost, 1e-3, 50, 1 ./ [100.0, 1, 1, 1, 1, 1, 25, 1])

# Main MCMC loop
out = mcmc(beta, kern, 10000, 20)

# Plot results
plot(1:10000, out, layout=(4, 2))
savefig("trace-hmc-ad.pdf")
histogram(out, layout=(4, 2))
savefig("hist-hmc-ad.pdf")
plot(1:400, autocor(out, 1:400), layout = (4, 2))
savefig("acf-hmc-ad.pdf")

# eof
