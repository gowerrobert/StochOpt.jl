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
options = set_options(max_iter=10^8, max_time=1000.0, max_epocs=50, force_continue=true, initial_point="zeros");
options.batchsize = 10;

## Load problem
datapath = "./data/";
data = "australian";
X, y = loadDataset(datapath, data);
prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-1, scaling="column-scaling");

## Running methods
OUTPUTS = [];  # List of saved outputs
#######
options.stepsize_multiplier = 1e-3;
options.batchsize = 1;
options.skip_error_calculation = 1000;
SVRG_nice = initiate_SVRG_nice(prob, options);
output = minimizeFunc(prob, SVRG_nice, options);
OUTPUTS = [OUTPUTS; output];
######
options.batchsize = 50;
options.skip_error_calculation = 500;
SVRG_nice = initiate_SVRG_nice(prob, options);
output = minimizeFunc(prob, SVRG_nice, options);
OUTPUTS = [OUTPUTS; output];
#######
options.batchsize = prob.numdata;
options.skip_error_calculation = 50;
SVRG_nice = initiate_SVRG_nice(prob, options);
output = minimizeFunc(prob, SVRG_nice, options);
OUTPUTS = [OUTPUTS; output];

## Saving outputs and plots
save_path = "./experiments/SVRG/";
if !isdir(save_path) # create directory if not existing
    if !isdir("./experiments/")
        mkdir("./experiments/");
    end
    mkdir(save_path);
    mkdir(string(save_path, "data/"));
    mkdir(string(save_path, "figures/"));
end

data_path = string(save_path, "data/");
if !isdir(data_path)
    mkdir(data_path);
end
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
savename = string(savename, "-", SVRG_nice.name);
save("$(data_path)$(savename).jld", "OUTPUTS", OUTPUTS);

if !isdir(string(save_path, "figures/"))
    mkdir(string(save_path, "figures/"));
end
pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, methodname=SVRG_nice.name, path=save_path) # Plot and save output
