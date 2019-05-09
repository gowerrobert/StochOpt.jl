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
options = set_options(max_iter=10^8, max_time=100.0, max_epocs=50, force_continue=true, initial_point="zeros", skip_error_calculation = 1000, repeat_stepsize_calculation=false)

## Debugging settings
# options = set_options(max_iter=10^8, max_time=0.005, max_epocs=1, force_continue=true, initial_point="zeros", skip_error_calculation = 1, repeat_stepsize_calculation=false)
# numinneriters = 5

## Load problem
datapath = "./data/"
data = "australian"
X, y = loadDataset(datapath, data)
prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-1, scaling="column-scaling")

## Running methods
OUTPUTS = [] # list of saved outputs

## Vanilla-SVRG with 1-nice sampling (m = 2n, b = 1, step size = gamma^*)
options.batchsize = 1
sampling = build_sampling("nice", prob.numdata, options)
options.stepsize_multiplier = -1.0 # 1/10Lmax
numinneriters = 2*prob.numdata # 2*n
SVRG_vanilla = initiate_SVRG_vanilla(prob, options, sampling, numinneriters=numinneriters)

println("-------------------- WARM UP --------------------")
options.max_iter = 3
minimizeFunc(prob, SVRG_vanilla, options) # Warm up
options.max_iter = 10^8
SVRG_vanilla.reset(prob, SVRG_vanilla, options)
println("-------------------------------------------------")

output1 = minimizeFunc(prob, SVRG_vanilla, options)
str_m_1 = @sprintf "%d" SVRG_vanilla.numinneriters
str_b_1 = @sprintf "%d" SVRG_vanilla.batchsize
str_step_1 = @sprintf "%.2e" SVRG_vanilla.stepsize
output1.name = latexstring("$(output1.name) \$(m = 2n = $str_m_1, b = $str_b_1 , \\gamma^* = $str_step_1\$)")
OUTPUTS = [OUTPUTS; output1]

## SVRG-Bubeck with b-nice sampling (m = m^*, b = 1, step size = gamma^*)
options.batchsize = 1
sampling = build_sampling("nice", prob.numdata, options)
options.stepsize_multiplier = -1.0 # Theoretical step size in boot_SVRG_bubeck
numinneriters = -1 # 20*Lmax/mu
SVRG_bubeck = initiate_SVRG_bubeck(prob, options, sampling, numinneriters=numinneriters)
output2 = minimizeFunc(prob, SVRG_bubeck, options)
str_m_2 = @sprintf "%d" SVRG_bubeck.numinneriters
str_b_2 = @sprintf "%d" SVRG_bubeck.batchsize
str_step_2 = @sprintf "%.2e" SVRG_bubeck.stepsize
output2.name = latexstring("$(output2.name) \$(m^* = $str_m_2, b = $str_b_2 , \\gamma^* = $str_step_2)\$")
OUTPUTS = [OUTPUTS; output2]

## Free-SVRG with b-nice sampling (m = m^*, b = b^*, step size = gamma^*)
options.batchsize = optimal_minibatch_free_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)
sampling = build_sampling("nice", prob.numdata, options)
options.stepsize_multiplier = -1.0 # Theoretical step size in boot_free_SVRG
numinneriters = -1 # (\cL + 2\rho)/mu
free_SVRG = initiate_free_SVRG(prob, options, sampling, numinneriters=numinneriters, averaged_reference_point=true)
output3 = minimizeFunc(prob, free_SVRG, options)
str_m_3 = @sprintf "%d" free_SVRG.numinneriters
str_b_3 = @sprintf "%d" free_SVRG.batchsize
str_step_3 = @sprintf "%.2e" free_SVRG.stepsize
output3.name = latexstring("$(output3.name) \$(m^* = $str_m_3, b^* = $str_b_3 , \\gamma^* = $str_step_3)\$")
OUTPUTS = [OUTPUTS; output3]

## Saving outputs and plots
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
savename = string(savename, "-", "demo_SVRG_algos_nice")
save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

pyplot() # gr() pyplot() # pgfplots() #plotly()
# plot_outputs_Plots(OUTPUTS, prob, options, methodname="demo_SVRG_algos_nice", path=save_path) # Plot and save output
plot_outputs_Plots(OUTPUTS, prob, options, methodname="demo_SVRG_algos_nice", path=save_path, legendfont=8) # Plot and save output