using JLD
using Plots
using StatsBase
using Match
using Combinatorics
using Random
using Printf
using LinearAlgebra
using Statistics
using Base64
include("./src/StochOpt.jl")

Random.seed!(1)

## Basic parameters and options for solvers
options = set_options(max_iter=10^8, max_time=100.0, max_epocs=50, force_continue=true, initial_point="zeros",
                      skip_error_calculation = 1000)

## Load problem
datapath = "./data/"
data = "australian"
X, y = loadDataset(datapath, data)
prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-1, scaling="column-scaling")

## Running methods
OUTPUTS = []  # List of saved outputs

# b = 1, m = n, step size = 1e-3
options.batchsize = 1
options.stepsize_multiplier = 1e-3

free_SVRG_nice1 = initiate_free_SVRG_nice(prob, options, averaged_reference_point=true)
free_SVRG_nice1.numinneriters = prob.numdata
free_SVRG_nice1.name = "$(free_SVRG_nice1.name) (m=n)"

output = minimizeFunc(prob, free_SVRG_nice1, options)
OUTPUTS = [OUTPUTS; output]


## b = 1, m = m^*, step size = 1e-3
options.batchsize = 1
options.stepsize_multiplier = 1e-3
# options.stepsize_multiplier = -1.0 # automatic step size in boot_free_SVRG_nice

free_SVRG_nice2 = initiate_free_SVRG_nice(prob, options, averaged_reference_point=false)
free_SVRG_nice2.name = "$(free_SVRG_nice2.name) (m=m^*)"
output = minimizeFunc(prob, free_SVRG_nice2, options)
OUTPUTS = [OUTPUTS; output]


## b = b^*, m = n, step size = 1e-3
options.batchsize = optimal_minibatch_free_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)

println("Theoretical mini-batch size: ", options.batchsize)
options.stepsize_multiplier = 1e-3
# options.stepsize_multiplier = -1.0 # automatic step size in boot_free_SVRG_nice

free_SVRG_nice3 = initiate_free_SVRG_nice(prob, options, averaged_reference_point=true)
free_SVRG_nice3.name = "$(free_SVRG_nice3.name) (b=b^*, m=m^*)"
# free_SVRG_nice3.numinneriters = floor(Int, (2*log(2)*(free_SVRG_nice3.expected_smoothness+2*free_SVRG_nice3.expected_residual)) / free_SVRG_nice3.mu)
println("Theoretical inner loop size: ", free_SVRG_nice3.numinneriters)
output = minimizeFunc(prob, free_SVRG_nice3, options)
OUTPUTS = [OUTPUTS; output]


## Saving outputs and plots
save_path = "./experiments/SVRG/"
if !isdir(save_path) # create directory if not existing
    if !isdir("./experiments/")
        mkdir("./experiments/")
    end
    mkdir(save_path)
    mkdir("$(save_path)data/")
    mkdir("$(save_path)figures/")
end

if !isdir("$(save_path)data/")
    mkdir("$(save_path)data/")
end
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
savename = string(savename, "-", "demo_free_SVRG_nice")
save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

if !isdir("$(save_path)figures/")
    mkdir("$(save_path)figures/")
end

pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, methodname="demo_free_SVRG_nice", path=save_path) # Plot and save output
