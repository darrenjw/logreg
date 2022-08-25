# Sample a logistic regression poster (Pima data) using the bouncy particle sampler
# with https://github.com/mschauer/ZigZagBoomerang.jl
using ZigZagBoomerang 
using ParquetFiles, DataFrames, Random, Distributions, StatsBase, ForwardDiff, Plots
using LinearAlgebra

# Load and process the data
df = DataFrame(load(joinpath(@__DIR__, "..", "pima.parquet")))
y = map((yi) -> yi == "Yes" ? 1.0 : 0.0, df.type)
X = hcat(ones(200, 1), Matrix(df[:,1:7]))

# Define target density/potential
function lprior(β)
    logpdf(Normal(0,10), β[1]) + sum(bi -> logpdf(Normal(0,1), bi), @view β[2:8])
end
ll(β, X, y) = sum(-log.(1.0 .+ exp.(-(2*y .- 1.0).*(X * β))))
ϕ(β, X, y) = -(lprior(β) + ll(β, X, y)) # potential

# Actually, we need a gradient... 
function ∇ϕ!(out, t, β, _, _,  x, y) # gradient of ϕ
    @. out = -β 
    out[1] = out[1]/100 # scale
    out .+= transpose(x) * (y .- (1.0 ./ (1.0 .+ exp.(-x * β))))
    out
end

# ... and directional derivatives we get with ForwardDiff
function dϕ(t, x, v, _, X, y) # two directional derivatives of ϕ
    u = ForwardDiff.derivative(t -> ϕ(x + t*v, X, y), ForwardDiff.Dual{:dϕ}(0.0, 1.0))
    u.value, u.partials[]
end

# Diagonal covariance estimate from warm-up (en lieu of mass matrix)
M = ZigZagBoomerang.PDMats.PDiagMat([3.5, 0.0038, 4.4e-5, 0.00036, 0.00052, 0.002, 0.33, 0.00048])

# Going to use BouncyParticle sampler 

Z = BouncyParticle(missing, missing, # ignored
    2.0, # momentum refreshment rate 
    0.9, # momentum correlation / only gradually change momentum in refreshment/momentum update
    M, # metric
    missing
) 

# Initialize sampler
d = 8 # number of parameters 
x0 = zeros(d) # starting point sampler
θ0 = randn(d) # starting direction sampler
T = 4000. # end time (similar to number of samples in MCMC)
bound = ZigZagBoomerang.LocalBound(20.0) # local bound on the rate of direction changes


# Call the sampler
trace, final, (acc, num), cs = @time pdmp(
        dϕ, # return first two directional derivatives of negative target log-density in direction v
        ∇ϕ!, # return gradient of negative target log-density
        0.0, x0, θ0, T, # initial state and duration
        bound, # inital guess for bound 
        Z, # sampler
        X, y; # data
        progress=true, # show progress bar
)
# Get time and location of sampling points seperately as two vectors
ts, xs = ZigZagBoomerang.sep(trace)

# Plots.jl wants matrix and not vector of vectors
out = [x[j] for x in xs, j in 1:d] 

# Plot results
plot(ts, out, layout=(4, 2))
savefig("tracebouncy.pdf")
histogram(out, layout=(4, 2))
savefig("histbouncy.pdf")
plot(1:400, autocor(out, 1:400), layout = (4, 2))
savefig("acfbouncy.pdf")