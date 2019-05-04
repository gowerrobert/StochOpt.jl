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
options = set_options(max_iter=10^8, max_time=1000.0, max_epocs=150, force_continue=true, initial_point="zeros");

## Load problem
datapath = "./data/";
data = "australian";
X, y = loadDataset(datapath, data);
prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-1, scaling="column-scaling");

## Testing step size grid search
options.batchsize = 1;
options.skip_error_calculation = 500;
options.repeat_stepsize_calculation = true; # Enforce the step size grid search
SAGA_nice = initiate_SAGA_nice(prob, options);
output = minimizeFunc_grid_stepsize(prob, SAGA_nice, options);

## Saving the associated plot
path = "./experiments/SAGA_nice/";
pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots([output], prob, options, methodname=SAGA_nice.name, path=path) # Plot and save output
