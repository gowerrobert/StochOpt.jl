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
save_path = "./experiments/Free_SVRG/"
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

## Settings to evaluate grossly the performance (error computed every 100 iterations, 30 epochs)
# options = set_options(max_iter=10^8, max_time=10000.0, max_epocs=30, initial_point="zeros", skip_error_calculation=1)

## Settings to monitor leaping (error computed at each iteration, very few epochs)
# options = set_options(max_iter=20, max_time=10.0^8, max_epocs=10^8, initial_point="zeros", skip_error_calculation=1)
options = set_options(max_iter=5, max_time=10.0^8, max_epocs=10^8, initial_point="zeros", skip_error_calculation=1)

## Load problem
datapath = "./data/"
data = "australian"     # n =     690, d = 22
# data = "ijcnn1_full"    # n = 141,691, d =
# data = "covtype_binary" # n = 581,012, d = 55
X, y = loadDataset(datapath, data)
prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-3, scaling="column-scaling")

## Running methods
OUTPUTS = [] # list of saved outputs

## Free-SVRG with b-nice sampling (b = 1, step sizes = alpha^*)
options.batchsize = 1 # prob.numdata
sampling = build_sampling("nice", prob.numdata, options)

idx_expe = 1

# for idx_expe=1:length(proba_grid)
    options.stepsize_multiplier = -1.0 # theoretical step sizes in boot_Leap_SVRG
    # options.skip_error_calculation = skip_errors[idx_expe]
    leap = initiate_Leap_SVRG(prob, options, sampling, proba)

    if idx_expe == 1
        println("-------------------- WARM UP --------------------")
        minimizeFunc(prob, leap, options) # Warm up
        # leap = initiate_Leap_SVRG(prob, options, sampling, proba)
        leap.reset(prob, leap, options)
        println("-------------------------------------------------")
    end

    output = minimizeFunc(prob, leap, options)
    str_b = @sprintf "%d" sampling.batchsize
    str_proba = @sprintf "%.3f" proba
    str_step_sto = @sprintf "%.2e" leap.stochastic_stepsize
    str_step_grad = @sprintf "%.2e" leap.gradient_stepsize
    output.name = latexstring("\$b = $str_b, p = $str_proba, \\eta^* = $str_step_grad, \\alpha^* = $str_step_sto \$")

    global OUTPUTS = [OUTPUTS; output]
# end

pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, methodname="debug_Leap_SVRG", path=save_path, legendfont=10) # Plot and save output