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
# options = set_options(max_iter=10^8, max_time=100.0, max_epocs=50, force_continue=true, initial_point="zeros", skip_error_calculation=1000, repeat_stepsize_calculation=false)

## Debugging settings
options = set_options(max_iter=10^8, max_time=0.005, max_epocs=1, force_continue=true, initial_point="zeros", skip_error_calculation = 1, repeat_stepsize_calculation=false)
numinneriters = 5

## Load problem
datapath = "./data/"
data = "australian"
X, y = loadDataset(datapath, data)
prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-1, scaling="column-scaling")

## Running methods
OUTPUTS = [] # list of saved outputs

## Vanilla-SVRG with 1-nice sampling (m = 2n, b = 1, step size = gamma^*)
options.batchsize = 1
sampling = build_sampling("independent", prob.numdata, options)
options.stepsize_multiplier = -1.0 # 1/10Lmax
numinneriters = 2*prob.numdata # 2*n
SVRG_vanilla = initiate_SVRG_vanilla(prob, options, sampling, numinneriters=numinneriters)

println("-------------------- WARM UP --------------------")
options.max_iter = 3
minimizeFunc(prob, SVRG_vanilla, options) # Warm up
options.max_iter = 10^8
println("-------------------------------------------------\n")

output = minimizeFunc(prob, SVRG_vanilla, options)
# str_m_1 = @sprintf "%d" SVRG_vanilla.numinneriters
# str_b_1 = @sprintf "%d" SVRG_vanilla.batchsize
# str_step_1 = @sprintf "%.2e" SVRG_vanilla.stepsize
# output.name = latexstring("Vanilla SVRG \$(m = 2n = $str_m_1, b = $str_b_1 , \\gamma^* = $str_step_1\$)")
OUTPUTS = [OUTPUTS; output]

## SVRG-Bubeck with b-nice sampling (m = m^*, b = 1, step size = gamma^*)
options.batchsize = 1
sampling = build_sampling("independent", prob.numdata, options)
options.stepsize_multiplier = -1.0 # Theoretical step size in boot_SVRG_bubeck
numinneriters = 2*prob.numdata # 2*n
SVRG_bubeck = initiate_SVRG_bubeck(prob, options, sampling, numinneriters=numinneriters)
output = minimizeFunc(prob, SVRG_bubeck, options)
# str_m_2 = @sprintf "%d" SVRG_bubeck.numinneriters
# str_b_2 = @sprintf "%d" SVRG_bubeck.batchsize
# str_step_2 = @sprintf "%.2e" SVRG_bubeck.stepsize
# output.name = latexstring("SVRG-Bubeck \$(m^* = $str_m_2, b = $str_b_2 , \\gamma^* = $str_step_2)\$")
OUTPUTS = [OUTPUTS; output]

## Free-SVRG with 1-uniform independent sampling (m = m^*, b = 1, step size = gamma^*)
options.batchsize = 1
sampling = build_sampling("independent", prob.numdata, options)
options.stepsize_multiplier = -1.0 # Theoretical step size in boot_Free_SVRG
numinneriters = 2*prob.numdata # 2*n
Free_SVRG = initiate_Free_SVRG(prob, options, sampling, numinneriters=numinneriters, averaged_reference_point=true)
output = minimizeFunc(prob, Free_SVRG, options)
# str_m_3 = @sprintf "%d" Free_SVRG.numinneriters
# str_b_3 = @sprintf "%d" Free_SVRG.batchsize
# str_step_3 = @sprintf "%.2e" Free_SVRG.stepsize
# output.name = latexstring("Free-SVRG \$(m^* = $str_m_3, b^* = $str_b_3 , \\gamma^* = $str_step_3)\$")
OUTPUTS = [OUTPUTS; output]


## Saving outputs and plots
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
savename = string(savename, "-", "demo_SVRG_algos_inde")
save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, methodname="demo_SVRG_algos_inde", path=save_path) # Plot and save output