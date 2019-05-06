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

## Path settings
#region
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

if !isdir("$(save_path)figures/")
    mkdir("$(save_path)figures/")
end
#endregion

Random.seed!(1)

## Basic parameters and options for solvers
options = set_options(max_iter=10^8, max_time=100.0, max_epocs=50, force_continue=true, initial_point="zeros", skip_error_calculation = 1000,
                      repeat_stepsize_calculation=false)

## Load problem
datapath = "./data/"
data = "australian"
X, y = loadDataset(datapath, data)
prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-1, scaling="column-scaling")

## Running methods
OUTPUTS = []  # List of saved outputs

## m = n, b = 1, step size = 1e-3
options.batchsize = 1
# options.stepsize_multiplier = 1e-3
free_SVRG_nice1 = initiate_free_SVRG_nice(prob, options, numinneriters=0, averaged_reference_point=true)
output = minimizeFunc_grid_stepsize(prob, free_SVRG_nice1, options)

str_m_1 = @sprintf "%d" free_SVRG_nice1.numinneriters
str_b_1 = @sprintf "%d" free_SVRG_nice1.batchsize
str_step_1 = @sprintf "%.2e" free_SVRG_nice1.stepsize
output.name = latexstring("\$m = n = $str_m_1, b = $str_b_1 , \\gamma_\\mathrm{grid search} = $str_step_1\$")
OUTPUTS = [OUTPUTS; output]

## m = m^*, b = 1, step size = 1e-3
options.batchsize = 1
options.stepsize_multiplier = 1e-3
free_SVRG_nice2 = initiate_free_SVRG_nice(prob, options, numinneriters=-1, averaged_reference_point=true)
output = minimizeFunc(prob, free_SVRG_nice2, options)

str_m_2 = @sprintf "%d" free_SVRG_nice2.numinneriters
str_b_2 = @sprintf "%d" free_SVRG_nice2.batchsize
str_step_2 = @sprintf "%.2e" free_SVRG_nice2.stepsize
output.name = latexstring("\$m^* = $str_m_2, b = $str_b_2 , \\gamma = $str_step_2\$")
OUTPUTS = [OUTPUTS; output]

## m = m^*, b = b^*, step size = gamma^*
options.batchsize = optimal_minibatch_free_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)
options.stepsize_multiplier = -1.0 # Theoretical step size in boot_free_SVRG_nice
free_SVRG_nice3 = initiate_free_SVRG_nice(prob, options, numinneriters=-1, averaged_reference_point=true)
output = minimizeFunc(prob, free_SVRG_nice3, options)

str_m_3 = @sprintf "%d" free_SVRG_nice3.numinneriters
str_b_3 = @sprintf "%d" free_SVRG_nice3.batchsize
str_step_3 = @sprintf "%.2e" free_SVRG_nice3.stepsize
output.name = latexstring("\$m^* = $str_m_3, b^* = $str_b_3 , \\gamma^* = $str_step_3\$")
OUTPUTS = [OUTPUTS; output]
println("Theoretical mini-batch size: ", options.batchsize)
println("Theoretical inner loop size: ", free_SVRG_nice3.numinneriters)

## Saving outputs and plots
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
savename = string(savename, "-", "demo_free_SVRG_nice")
save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, methodname="demo_free_SVRG_nice", path=save_path) # Plot and save output