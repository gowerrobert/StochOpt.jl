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

## Basic parameters and options for solvers
options = set_options(max_iter=10^8, max_time=0.01, max_epocs=1, force_continue=true, initial_point="zeros");

## Load problem
datapath = "./data/";
data = "australian";
X, y = loadDataset(datapath, data);
prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-1, scaling="column-scaling");

## Running methods
OUTPUTS = [];  # List of saved outputs
#######
# options.stepsize_multiplier = 1e-3;
# options.batchsize = 1;
# options.skip_error_calculation = 1;
# SVRG_nice = initiate_SVRG_nice(prob, options);
# output = minimizeFunc(prob, SVRG_nice, options);
# OUTPUTS = [OUTPUTS; output];
######
# options.batchsize = optimal_minibatch_free_SVRG(n, mu, L, Lmax);
options.batchsize = 1;
println("Theoretical mini-batch size: ", options.batchsize);

options.stepsize_multiplier = -1.0; # automatic step size in boot_SVRG_nice
options.skip_error_calculation = 1;
SVRG_nice = initiate_SVRG_nice(prob, options);

# SVRG_nice.numinneriters = floor(Int, (2*log(2)*(SVRG_nice.expected_smoothness+2*SVRG_nice.expected_residual)) / SVRG_nice.mu);
println("Theoretical inner loop size: ", SVRG_nice.numinneriters);

output = minimizeFunc(prob, SVRG_nice, options);


######
# options.batchsize = 50;
# options.skip_error_calculation = 500;
# SVRG_nice = initiate_SVRG_nice(prob, options);
# output = minimizeFunc(prob, SVRG_nice, options);
# OUTPUTS = [OUTPUTS; output];
# #######
# options.batchsize = prob.numdata;
# options.skip_error_calculation = 50;
# SVRG_nice = initiate_SVRG_nice(prob, options);
# output = minimizeFunc(prob, SVRG_nice, options);
# OUTPUTS = [OUTPUTS; output];

# ## Saving outputs and plots
# save_path = "./experiments/SVRG/";
# if !isdir(save_path) # create directory if not existing
#     if !isdir("./experiments/")
#         mkdir("./experiments/");
#     end
#     mkdir(save_path);
#     mkdir(string(save_path, "data/"));
#     mkdir(string(save_path, "figures/"));
# end

# data_path = string(save_path, "data/");
# if !isdir(data_path)
#     mkdir(data_path);
# end
# savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
# savename = string(savename, "-", SVRG_nice.name);
# save("$(data_path)$(savename).jld", "OUTPUTS", OUTPUTS);

# if !isdir(string(save_path, "figures/"))
#     mkdir(string(save_path, "figures/"));
# end
# pyplot() # gr() pyplot() # pgfplots() #plotly()
# plot_outputs_Plots(OUTPUTS, prob, options, methodname=SVRG_nice.name, path=save_path) # Plot and save output
