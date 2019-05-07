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
OUTPUTS = [] # list of saved outputs

## Vanilla-SVRG with 1-nice sampling (m = 2n, b = 1, step size = gamma^*)
options.batchsize = 1
options.stepsize_multiplier = -1.0 # 1/10Lmax
SVRG_vanilla = initiate_SVRG_vanilla(prob, options, "nice", numinneriters=2*prob.numdata) # 2n
output = minimizeFunc(prob, SVRG_vanilla, options)

str_m_1 = @sprintf "%d" SVRG_vanilla.numinneriters
str_b_1 = @sprintf "%d" SVRG_vanilla.batchsize
str_step_1 = @sprintf "%.2e" SVRG_vanilla.stepsize
output.name = latexstring("Vanilla SVRG \$(m = 2n = $str_m_1, b = $str_b_1 , \\gamma^* = $str_step_1\$)")
OUTPUTS = [OUTPUTS; output]

## Free-SVRG with b-nice sampling (m = m^*, b = b^*, step size = gamma^*)
options.batchsize = optimal_minibatch_free_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)
options.stepsize_multiplier = -1.0 # Theoretical step size in boot_free_SVRG
free_SVRG = initiate_free_SVRG(prob, options, "nice", numinneriters=-1, averaged_reference_point=true)
output = minimizeFunc(prob, free_SVRG, options)

str_m_2 = @sprintf "%d" free_SVRG.numinneriters
str_b_2 = @sprintf "%d" free_SVRG.batchsize
str_step_2 = @sprintf "%.2e" free_SVRG.stepsize
output.name = latexstring("Free-SVRG \$(m^* = $str_m_2, b^* = $str_b_2 , \\gamma^* = $str_step_2)\$")
OUTPUTS = [OUTPUTS; output]

## Saving outputs and plots
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
savename = string(savename, "-", "demo_SVRG_algos_nice")
save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, methodname="demo_SVRG_algos_nice", path=save_path) # Plot and save output