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
options = set_options(max_iter=10^8, max_time=100.0, max_epocs=50, force_continue=false, initial_point="zeros", skip_error_calculation=10, repeat_stepsize_calculation=false)
# options = set_options(max_iter=10^8, max_time=0.001, max_epocs=50, force_continue=false, initial_point="zeros", skip_error_calculation=1, repeat_stepsize_calculation=false)

## Load problem
datapath = "./data/"
data = "australian"
X, y = loadDataset(datapath, data)
prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-1, scaling="column-scaling")

## Running methods
OUTPUTS = []  # List of saved outputs

## Sampling procedure
sampling = "nice"
sampling = "independent"
b = 100
unif_inde_proba = (b/prob.numdata)*ones(prob.numdata) # uniform independent probabilities: p_i = 1/n

## m = n, b = 1, step size = 1e-3
options.batchsize = 100
free_SVRG1 = initiate_free_SVRG(prob, options, sampling, numinneriters=0, averaged_reference_point=true, probs=unif_inde_proba)
options.stepsize_multiplier, = get_saved_stepsize(prob.name, free_SVRG1.name, options)
if options.stepsize_multiplier == 0.0 || options.repeat_stepsize_calculation
    output = minimizeFunc_grid_stepsize(prob, free_SVRG1, options)
    step_practical_gridsearch, = get_saved_stepsize(prob.name, free_SVRG1.name, options)
end
# options.stepsize_multiplier = 1e-3
output = minimizeFunc(prob, free_SVRG1, options)

str_m_1 = @sprintf "%d" free_SVRG1.numinneriters
str_b_1 = @sprintf "%d" free_SVRG1.batchsize
str_step_1 = @sprintf "%.2e" free_SVRG1.stepsize
output.name = latexstring("\$m = n = $str_m_1, b = $str_b_1 , \\gamma_\\mathrm{grid search} = $str_step_1\$")
OUTPUTS = [OUTPUTS; output]

## m = m^*, b = 1, step size = 1e-3
options.batchsize = 100
options.stepsize_multiplier = 1e-3
free_SVRG2 = initiate_free_SVRG(prob, options, sampling, numinneriters=-1, averaged_reference_point=true, probs=unif_inde_proba)
output = minimizeFunc(prob, free_SVRG2, options)

str_m_2 = @sprintf "%d" free_SVRG2.numinneriters
str_b_2 = @sprintf "%d" free_SVRG2.batchsize
str_step_2 = @sprintf "%.2e" free_SVRG2.stepsize
output.name = latexstring("\$m^* = $str_m_2, b = $str_b_2 , \\gamma = $str_step_2\$")
OUTPUTS = [OUTPUTS; output]

## m = m^*, b = b^*, step size = gamma^*
options.batchsize = 100 #optimal_minibatch_free_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)
options.stepsize_multiplier = -1.0 # Theoretical step size in boot_free_SVRG
free_SVRG3 = initiate_free_SVRG(prob, options, sampling, numinneriters=-1, averaged_reference_point=true, probs=unif_inde_proba)
output = minimizeFunc(prob, free_SVRG3, options)

str_m_3 = @sprintf "%d" free_SVRG3.numinneriters
str_b_3 = @sprintf "%d" free_SVRG3.batchsize
str_step_3 = @sprintf "%.2e" free_SVRG3.stepsize
output.name = latexstring("\$m^* = $str_m_3, b^* = $str_b_3 , \\gamma^* = $str_step_3\$")
OUTPUTS = [OUTPUTS; output]
println("Theoretical mini-batch size: ", options.batchsize)
println("Theoretical inner loop size: ", free_SVRG3.numinneriters)

## Saving outputs and plots
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
savename = string(savename, "-", "demo_free_SVRG")
save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, methodname="demo_free_SVRG", path=save_path) # Plot and save output