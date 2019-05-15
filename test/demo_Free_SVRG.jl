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
# options = set_options(max_iter=10^8, max_time=10.0^4, max_epocs=50, force_continue=false, initial_point="zeros", skip_error_calculation=100, repeat_stepsize_calculation=false, rep_number=3)


## Settings to monitor leaping (error computed at each iteration, very few epochs)
# options = set_options(max_iter=20, max_time=10.0^8, max_epocs=10^8, initial_point="zeros", skip_error_calculation = 1)
options = set_options(max_iter=10^8, max_time=10.0^8, max_epocs=10, initial_point="zeros", skip_error_calculation = 5000, repeat_stepsize_calculation=true, rep_number=3)

## Load problem
datapath = "./data/"
# data = "australian"     # n =     690, d = 15
data = "ijcnn1_full"    # n = 141,691, d = 23
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

## m = n, b = b^*(n), step size = gamma^*(b^*(n))
# options.skip_error_calculation = 10

numinneriters = prob.numdata
if numinneriters == prob.numdata
    options.batchsize = optimal_minibatch_Free_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)
else
    options.batchsize = 1 # default value for other inner loop sizes
end
sampling = build_sampling("nice", prob.numdata, options)
f1 = initiate_Free_SVRG(prob, options, sampling, numinneriters=numinneriters, averaged_reference_point=true)
options.stepsize_multiplier = -1.0 # Theoretical step size in boot_Free_SVRG

println("-------------------- WARM UP --------------------")
old_max_iter = options.max_iter
options.max_iter = 30
minimizeFunc(prob, f1, options) # Warm up
f1.reset(prob, f1, options)
options.max_iter = old_max_iter
println("-------------------------------------------------")

output = minimizeFunc(prob, f1, options)
str_m_1 = @sprintf "%d" f1.numinneriters
str_b_1 = @sprintf "%d" f1.batchsize
str_step_1 = @sprintf "%.2e" f1.stepsize
output.name = latexstring("\$m = n = $str_m_1, b^*(n) = $str_b_1, \\gamma^* (b^*(n)) = $str_step_1\$")
OUTPUTS = [OUTPUTS; output]
println("Theoretical optimal mini-batch size: ", f1.batchsize)


## m = n, b = b^*(n), step size = grid searched
# options.skip_error_calculation = 5000

numinneriters = prob.numdata
if numinneriters == prob.numdata
    options.batchsize = optimal_minibatch_Free_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)
else
    options.batchsize = 1 # default value for other inner loop sizes
end

sampling = build_sampling("nice", prob.numdata, options)
f_grid = initiate_Free_SVRG(prob, options, sampling, numinneriters=numinneriters, averaged_reference_point=true)
options.stepsize_multiplier, = get_saved_stepsize(prob.name, f_grid.name, options)
if options.stepsize_multiplier == 0.0 || options.repeat_stepsize_calculation
    output = minimizeFunc_grid_stepsize(prob, f_grid, options)
    step_practical_gridsearch, = get_saved_stepsize(prob.name, f_grid.name, options)
end
output = minimizeFunc(prob, f_grid, options)
str_m_2 = @sprintf "%d" f_grid.numinneriters
str_b_2 = @sprintf "%d" f_grid.batchsize
str_step_2 = @sprintf "%.2e" f_grid.stepsize
output.name = latexstring("\$m = n = $str_m_2, b^*(n) = $str_b_2, \\gamma_\\mathrm{grid search} = $str_step_2\$")
OUTPUTS = [OUTPUTS; output]

## m = n/b, b = 10, step size = gamma^*(b)
options.skip_error_calculation = 500

options.batchsize = 10
numinneriters = round(Int64, prob.numdata/options.batchsize)

sampling = build_sampling("nice", prob.numdata, options)
f3 = initiate_Free_SVRG(prob, options, sampling, numinneriters=numinneriters, averaged_reference_point=true)
options.stepsize_multiplier = -1.0 # Theoretical step size in boot_Free_SVRG
output = minimizeFunc(prob, f3, options)
str_m_3 = @sprintf "%d" f3.numinneriters
str_b_3 = @sprintf "%d" f3.batchsize
str_step_3 = @sprintf "%.2e" f3.stepsize
output.name = latexstring("\$m = n/b = $str_m_3, b = $str_b_3, \\gamma^* (b) = $str_step_3\$")
OUTPUTS = [OUTPUTS; output]

## Saving outputs and plots
# savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
# savename = string(savename, "-", "demo_Free_SVRG")
# save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

pyplot() # gr() pyplot() # pgfplots() #plotly()
# plot_outputs_Plots(OUTPUTS, prob, options, methodname="demo_Free_SVRG", path=save_path, legendfont=10) # Plot and save output
plot_outputs_Plots(OUTPUTS, prob, options, methodname="debug_Free_SVRG", path=save_path, legendfont=10) # Plot and save output