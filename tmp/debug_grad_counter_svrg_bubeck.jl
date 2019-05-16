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
include("../src/StochOpt.jl")

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
options = set_options(max_iter=10^8, max_time=100.0, max_epocs=50, force_continue=true, initial_point="zeros", skip_error_calculation = 100, repeat_stepsize_calculation=false)
numinneriters = -1

## Debugging settings
# options = set_options(max_iter=10^8, max_time=10.0^8, max_epocs=3, force_continue=true, initial_point="zeros", skip_error_calculation = 1, repeat_stepsize_calculation=false)
# numinneriters = 5

## Load problem
datapath = "./data/"
data = "australian"
X, y = loadDataset(datapath, data)
prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-3, scaling="column-scaling")

## Running methods
OUTPUTS = [] # list of saved outputs

## SVRG-Bubeck with b-nice sampling (m, b = 1, step size = gamma^*)
options.batchsize = 1
sampling = build_sampling("nice", prob.numdata, options)
options.stepsize_multiplier = -1.0 # Theoretical step size in boot_SVRG_bubeck
bubeck = initiate_SVRG_bubeck(prob, options, sampling, numinneriters=numinneriters)

println("-------------------- WARM UP --------------------")
minimizeFunc(prob, bubeck, options) # Warm up
# bubeck = initiate_SVRG_bubeck(prob, options, sampling, numinneriters=numinneriters)
bubeck.reset(prob, bubeck, options)
println("-------------------------------------------------")

output2 = minimizeFunc(prob, bubeck, options)
str_m_2 = @sprintf "%d" bubeck.numinneriters
str_b_2 = @sprintf "%d" bubeck.batchsize
str_step_2 = @sprintf "%.2e" bubeck.stepsize
output2.name = latexstring("$(output2.name) \$(m = $str_m_2, b = $str_b_2 , \\gamma^* = $str_step_2)\$")
OUTPUTS = [OUTPUTS; output2]

pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, methodname="debug_Bubeck_grad_counter", path=save_path, legendfont=10) # Plot and save output