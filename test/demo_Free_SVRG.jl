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

## Basic parameters and options for solvers
options = set_options(max_iter=10^8, max_time=50.0, max_epocs=50, force_continue=false, initial_point="zeros", skip_error_calculation=1, repeat_stepsize_calculation=false, rep_number=3)

## Load problem
datapath = "./data/"
data = "australian"     # n =     690, d = 22
# data = "ijcnn1_full"    # n = 141,691, d =
# data = "covtype_binary" # n = 581,012, d = 55
X, y = loadDataset(datapath, data)
prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-3, scaling="column-scaling")

## Running methods
OUTPUTS = []  # List of saved outputs

## Sampling procedure
#region
# b-nice sampling
# options.batchsize = 1
# sampling = build_sampling("nice", prob.numdata, options)
# sampling.sampleindices(sampling)
# sampling.batchsize
# sampling.name

# uniform b-independent sampling
# options.batchsize = 1
# sampling = build_sampling("independent", prob.numdata, options)
# sampling.sampleindices(sampling)
# sampling.batchsize
# sampling.name
#endregion

## m = m^*, b = b^*(m^*), step size = gamma^*(b^*)
options.skip_error_calculation = 1000
options.batchsize = optimal_minibatch_Free_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)
sampling = build_sampling("nice", prob.numdata, options)
f1 = initiate_Free_SVRG(prob, options, sampling, numinneriters=prob.numdata, averaged_reference_point=true)
options.stepsize_multiplier = -1.0 # Theoretical step size in boot_Free_SVRG

println("-------------------- WARM UP --------------------")
options.max_iter = 3
minimizeFunc(prob, f1, options) # Warm up
f1.reset(prob, f1, options)
options.max_iter = 10^8
println("-------------------------------------------------")

output = minimizeFunc(prob, f1, options)
str_m_1 = @sprintf "%d" f1.numinneriters
str_b_1 = @sprintf "%d" f1.batchsize
str_step_1 = @sprintf "%.2e" f1.stepsize
output.name = latexstring("\$m^* = $str_m_1, b^* = $str_b_1, \\gamma^* = $str_step_1\$")
OUTPUTS = [OUTPUTS; output]
println("Theoretical mini-batch size: ", f1.batchsize)
println("Theoretical inner loop size: ", f1.numinneriters)

## m = n, b = b^*(m^*), step size = grid searched
# options.skip_error_calculation = 1000
# options.batchsize = optimal_minibatch_Free_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax) # m = n
# sampling = build_sampling("nice", prob.numdata, options)
# f_grid = initiate_Free_SVRG(prob, options, sampling, numinneriters=prob.numdata, averaged_reference_point=true)
# options.stepsize_multiplier, = get_saved_stepsize(prob.name, f_grid.name, options)
# if options.stepsize_multiplier == 0.0 || options.repeat_stepsize_calculation
#     output = minimizeFunc_grid_stepsize(prob, f_grid, options)
#     step_practical_gridsearch, = get_saved_stepsize(prob.name, f_grid.name, options)
# end
# output = minimizeFunc(prob, f_grid, options)
# str_m_2 = @sprintf "%d" f_grid.numinneriters
# str_b_2 = @sprintf "%d" f_grid.batchsize
# str_step_2 = @sprintf "%.2e" f_grid.stepsize
# output.name = latexstring("\$m^* = $str_m_2, b^* = $str_b_2, \\gamma_\\mathrm{grid search} = $str_step_2\$")
# OUTPUTS = [OUTPUTS; output]

## m = n, b = b(n), step size = gamma (corresponding to b=1)
options.skip_error_calculation = 5000
options.batchsize = minibatch_Free_SVRG_nice(prob.numdata, prob.numdata, prob.mu, prob.L, prob.Lmax)
sampling = build_sampling("nice", prob.numdata, options)
f_heuristic = initiate_Free_SVRG(prob, options, sampling, numinneriters=prob.numdata, averaged_reference_point=true)
options.stepsize_multiplier = -1.0 # Theoretical step size in boot_Free_SVRG
output = minimizeFunc(prob, f_heuristic, options)
str_m_3 = @sprintf "%d" f_heuristic.numinneriters
str_b_3 = @sprintf "%d" f_heuristic.batchsize
str_step_3 = @sprintf "%.2e" f_heuristic.stepsize
output.name = latexstring("\$m = n = $str_m_3, b(n) = $str_b_3, \\gamma = $str_step_3\$")
OUTPUTS = [OUTPUTS; output]

## Saving outputs and plots
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
savename = string(savename, "-", "demo_Free_SVRG")
save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, methodname="demo_Free_SVRG", path=save_path, legendfont=10) # Plot and save output