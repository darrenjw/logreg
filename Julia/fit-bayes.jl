#!/usr/bin/env julia
# fit-bayes.jl
# Fit Bayesian logistic regression using RW MH MCMC in Julia

# Should I use Julia? Maybe, maybe not...  https://yuri.is/not-julia/


using ParquetFiles, DataFrames, Random, Distributions, Plots, StatsBase
using LinearAlgebra: mul!
using LoopVectorization: @turbo

# Define some functions

function lprior(beta)
    d = Normal(0,1)
    logpdf(Normal(0,10), beta[1]) + sum(bi -> logpdf(d, bi), @view beta[2:8])
end

function ll!(buff, beta, x, y)
    mul!(buff, x, beta)
    T = promote_type(float(eltype(buff)), float(eltype(y)))
    if T <: Base.IEEEFloat && axes(buff) === axes(y)
        s = zero(T)
        @turbo for i in eachindex(buff)
            sy = y[i]
            sb = buff[i]
            s -= log(1.0 + exp(-(2*sy - 1.0)*sb))
        end
        s
    else
        sum(zip(y, buff)) do (sy, sb)
            -log(1.0 + exp(-(2*sy - 1.0)*sb))
        end
    end
end

lpost!(beta, (buff, x, y)) = lprior(beta) + ll!(buff, beta, x, y)

function mhKernel(logPost, rprop, rng)
    let logPost = logPost, rprop = rprop, rng = rng
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
    end
end

function mcmc(init, kernel, iters, thin)
    p = length(init)
    ll = -Inf
    mat = zeros(iters, p)
    x = init
    for i in 1:iters
        if iszero(i % 100)
            print(i); print(" ")
        end
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
df = DataFrame(load(joinpath(@__DIR__, "..", "pima.parquet")))
y = df.type
y = map((yi) -> yi == "Yes" ? 1.0 : 0.0, y)
x = df[:,1:7]
x = Matrix(x)
x = hcat(ones(200, 1), x)
# Set up for doing MCMC
beta = zeros(8)
beta[1] = -10
rng = MersenneTwister(1234)
norm = Normal(0, 0.02)
kern = let rng = rng, norm = norm
    fun = let buffer = similar(x, 8)
        x -> begin
            buffer[1] = x[1]+rand(rng, norm)*10
            rand!(rng, norm, @view buffer[2:8])
            @views buffer[2:8] .+= x[2:8]
            copy(buffer)
        end
    end
    mhKernel(Base.Fix2(lpost!, (similar(beta, axes(x, 1)), x, y)), fun, rng)
end
# Main MCMC loop
out = @time mcmc(beta, kern, 10000, 1000)

# Plot results
plot(1:10000, out, layout=(4, 2))
savefig("trace.pdf")
histogram(out, layout=(4, 2))
savefig("hist.pdf")
plot(1:400, autocor(out, 1:400), layout = (4, 2))
savefig("acf.pdf")


# eof
