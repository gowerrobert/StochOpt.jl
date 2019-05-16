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
save_path = "./experiments/L_SVRG_D/"
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
# options = set_options(max_iter=10^8, max_time=10000.0, max_epocs=50, force_continue=false, initial_point="zeros", skip_error_calculation=1000)
options = set_options(max_iter=10^8, max_time=10.0^5, max_epocs=10, force_continue=false, initial_point="zeros", skip_error_calculation=1)

## Load problem
datapath = "./data/"
data = "australian"
# data = "ijcnn1_full"    # n = 141,691, d =
X, y = loadDataset(datapath, data)
prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-3, scaling="column-scaling")

## Running methods
OUTPUTS = []  # List of saved outputs

proba_grid = [0.1]
# proba_grid = [0.1, 0.01, 0.005, 1/prob.numdata]

## L_SVRG_D with b-nice sampling (b = 1, proba= several values, step sizes = gamma^*)
options.batchsize = 1 # prob.numdata
sampling = build_sampling("nice", prob.numdata, options)

for idx_expe=1:length(proba_grid)
    proba = proba_grid[idx_expe]
    options.stepsize_multiplier = -1.0 # theoretical step sizes in boot_Leap_SVRG
    decreasing = initiate_L_SVRG_D(prob, options, sampling, proba)

    if idx_expe == 1
        println("-------------------- WARM UP --------------------")
        tmp = options.max_epocs
        options.max_epocs = 2
        minimizeFunc(prob, decreasing, options) # Warm up
        options.max_epocs = tmp
        decreasing.reset(prob, decreasing, options)
        println("-------------------------------------------------")
    end

    output = minimizeFunc(prob, decreasing, options)
    str_b = @sprintf "%d" sampling.batchsize
    str_proba = @sprintf "%.3f" proba
    str_step = @sprintf "%.2e" decreasing.stepsize
    output.name = latexstring("L_SVRG_D \$b = $str_b, p = $str_proba, \\gamma^* = $str_step\$")

    global OUTPUTS = [OUTPUTS; output]
end

pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, methodname="debug_L_SVRG_D", path=save_path) # Plot and save output

# ## nice sampling with p = 1/n
# options.batchsize = 1
# sampling = build_sampling("nice", prob.numdata, options)
# proba = 0.5
# # proba = 1/prob.numdata
# L_SVRG_D_1 = initiate_L_SVRG_D(prob, options, sampling, proba)
# options.stepsize_multiplier = -1.0 # Theoretical step size in boot_L_SVRG_D

# println("-------------------- WARM UP --------------------")
# options.max_iter = 3
# minimizeFunc(prob, L_SVRG_D_1, options) # Warm up
# options.max_iter = 10^8
# L_SVRG_D_1.reset(prob, L_SVRG_D_1, options)
# println("-------------------------------------------------")

# output = minimizeFunc(prob, L_SVRG_D_1, options)
# str_b_1 = @sprintf "%d" sampling.batchsize
# str_proba_1 = @sprintf "%.3f" proba
# str_step_1 = @sprintf "%.2e" L_SVRG_D_1.stepsize
# output.name = latexstring("\$b = $str_b_1, p = $str_proba_1 , \\gamma^* = $str_step_1\$")
# OUTPUTS = [OUTPUTS; output]


# ## nice sampling with p = 1/10, step size = grid searched
# options.batchsize = 1
# sampling = build_sampling("nice", prob.numdata, options)
# proba_3 = 0.99 # 1/sqrt(prob.numdata)
# L_SVRG_D_3 = initiate_L_SVRG_D(prob, options, sampling, proba_3)
# options.stepsize_multiplier = -1.0 # Theoretical step size in boot_L_SVRG_D
# output = minimizeFunc(prob, L_SVRG_D_3, options)
# str_b_3 = @sprintf "%d" sampling.batchsize
# str_proba_3 = @sprintf "%.3f" proba_3
# str_step_3 = @sprintf "%.2e" L_SVRG_D_3.stepsize
# output.name = latexstring("\$b = $str_b_3, p = $str_proba_3, \\gamma^* = $str_step_3\$")
# OUTPUTS = [OUTPUTS; output]

# ## nice sampling with p = 1/n, step size = grid searched
# options.batchsize = 1
# sampling = build_sampling("nice", prob.numdata, options)
# proba_2 = 1/prob.numdata
# L_SVRG_D_2 = initiate_L_SVRG_D(prob, options, sampling, proba_2)
# options.stepsize_multiplier, = get_saved_stepsize(prob.name, L_SVRG_D_2.name, options)
# if options.stepsize_multiplier == 0.0 || options.repeat_stepsize_calculation
#     output = minimizeFunc_grid_stepsize(prob, L_SVRG_D_2, options)
#     step_practical_gridsearch, = get_saved_stepsize(prob.name, L_SVRG_D_2.name, options)
# end
# output = minimizeFunc(prob, L_SVRG_D_2, options)
# str_b_2 = @sprintf "%d" sampling.batchsize
# str_proba_2 = @sprintf "%.3f" proba_2
# str_step_2 = @sprintf "%.2e" L_SVRG_D_2.stepsize
# output.name = latexstring("\$b = $str_b_2, p = $str_proba_2, \\gamma_\\mathrm{grid search} = $str_step_2\$")
# OUTPUTS = [OUTPUTS; output]


## Saving outputs and plots
# savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
# savename = string(savename, "-", "demo_L_SVRG_D")
# save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

# pyplot() # gr() pyplot() # pgfplots() #plotly()
# plot_outputs_Plots(OUTPUTS, prob, options, methodname="debug_L_SVRG_D", path=save_path) # Plot and save output