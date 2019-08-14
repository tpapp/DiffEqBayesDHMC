#####
##### Self-contained example to dissect broken tests in
##### https://github.com/JuliaDiffEq/DiffEqBayes.jl/blob/e033307768892e2f7242ae0aab3e09ec4819c11b/test/dynamicHMC.jl
#####
##### NOTE: instantiate and activate this project and cd() into its directory

####
#### part that just relies on the DiffEq ecosystem
####

using OrdinaryDiffEq, ParameterizedFunctions, RecursiveArrayTools, Parameters,
    Distributions, Random

struct DynamicHMCPosterior{TA,TP,TL,TR,TS,TK}
    alg::TA
    problem::TP
    likelihood::TL
    priors::TR
    σ_prior::TS
    kwargs::TK
end

function (P::DynamicHMCPosterior)(θ)
    @unpack a, σ = θ
    @unpack alg, problem, likelihood, priors, σ_prior, kwargs = P
    prob = remake(problem, u0 = convert.(eltype(a), problem.u0), p = a)
    sol = solve(prob, alg; kwargs...)
    if any((s.retcode != :Success for s in sol)) && any((s.retcode != :Terminated for s in sol))
        return -Inf
    end
    likelihood(sol, σ) + mapreduce(logpdf, +, priors, θ)
end

function data_log_likelihood(solution, data, t, σ)
    sum(sum(logpdf.(Normal.(0.0, σ), solution(t) .- data[:, i])) for (i, t) in enumerate(t))
end

function dynamic_hmc_posterior(alg, problem, data, t, priors, σ_prior; kwargs...)
    DynamicHMCPosterior(alg, problem,
                        (solution, σ) -> data_log_likelihood(solution, data, t, σ),
                        priors, σ_prior, kwargs)
end

####
#### ODE setup and data generation
####

Random.seed!(1)

f1 = @ode_def LotkaVolterraTest1 begin
  dx = a*x - x*y
  dy = -3*y + x*y
end a

a₀ = 1.5                        # true parameter
p = [a₀]
u0 = [1.0, 1.0]
tspan = (0.0,10.0)
prob1 = ODEProblem(f1, u0, tspan, p)
σ = 0.01                                 # noise, fixed for now
t = collect(range(1, stop=10,length=10)) # observation times
sol = solve(prob1, Tsit5())
randomized = VectorOfArray([(sol(t[i]) + σ * randn(2)) for i in 1:length(t)])
data = convert(Array, randomized)

P = dynamic_hmc_posterior(Tsit5(), prob1, data, t, (Normal(a₀, 0.1), ), InverseGamma(3,2);
                          maxiters = 10^5)

P((a = 1.5, σ = [0.01, 0.01]))                  # make sure log posterior works

####
#### problem setup in the LogDensityProblems API
####

using LogDensityProblems
using LogDensityProblems: logdensity, logdensity_and_gradient
using PGFPlotsX                 # plotting
import ForwardDiff              # AD
using TransformVariables

trans = as((a = asℝ₊, σ = as(Vector, asℝ₊, 2)))
ℓ = TransformedLogDensity(trans, P)
∇ℓ = ADgradient(:ForwardDiff, ℓ)

####
#### DynamicHMC & diagnostics
####

using DynamicHMC

###
### running mcmc
###

result = mcmc_with_warmup(Random.GLOBAL_RNG, ∇ℓ, 1000)

###
### diagnostics
###

using DynamicHMC.Diagnostics
EBFMI(result.tree_statistics)
summarize_tree_statistics(result.tree_statistics)

estimated = trans.(result.chain)

mean(x -> x.a, estimated)
mean(x -> x.σ[1], estimated)
mean(x -> x.σ[2], estimated)
