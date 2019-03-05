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

include("./src/StochOpt.jl") # Be carefull about the path here

default_path = "./data/";

Random.seed!(1);

### LOADING DATA ###
println("--- Loading data ---");
datasets = readlines("$(default_path)available_datasets.txt");

## Only loading datasets, no data generation
idx = 12; # ijcnn1_full
data = datasets[idx];
# scaling = "none";
scaling = "column-scaling";
# lambda = -1;
# lambda = 10^(-3);
lambda = 10^(-1);

@time X, y = loadDataset(default_path, data);

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-1), max_iter=10^8, max_time=10.0^2, max_epocs=10^8,
                    #   regularizor_parameter = "1/num_data", # fixes lambda
                      regularizor_parameter = "normalized",
                    #   regularizor_parameter = "Lbar/n",
                      initial_point="zeros", # is fixed not to add more randomness
                      force_continue=false); # force continue if diverging or if tolerance reached

@time prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling=scaling);

########################################### ijcnn1_full ############################################
#region
## Computing the solution with a serial gridsearch
@time get_fsol_logistic!(prob)

## BFGS, step = 2.0^(-1.0), 200 epochs
## BFGS-a-141691.0-0.01: step = 2^1.0, 200 epochs
## fsol = 0.19496891145648523
#endregion
######################################################################################################
