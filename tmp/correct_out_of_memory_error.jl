using JLD
using Plots
using StatsBase
using Match
using Combinatorics
using Random # julia 0.7
using Printf # julia 0.7
using LinearAlgebra # julia 0.7
using Statistics # julia 0.7
using Base64 # julia 0.7

# using Distributed
# addprocs(4)

include("./src/StochOpt.jl") # Be carefull about the path here

default_path = "./data/";

Random.seed!(1);

### LOADING DATA ###
println("--- Loading data ---");
# Available datasets are in "./data/available_datasets.txt" 
# datasets = ["fff", "gauss-5-8-0.0_seed-1234", "YearPredictionMSD", "abalone", "housing"];
datasets = readlines("$(default_path)available_datasets.txt");
#, "letter_scale", "heart", "phishing", "madelon", "a9a",
# "mushrooms", "phishing", "w8a", "gisette_scale",

## Only loading datasets, no data generation
idx = 14; # news20.binary
data = datasets[idx];

memory = Sys.free_memory() / 2^20;
X, y = loadDataset(data);
println("Occupied memory: ", (Sys.total_memory() - Sys.free_memory()) / 2^20, " MiB")

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-1), max_iter=10^8, max_time=10.0^2, max_epocs=10^8,
                    #   regularizor_parameter = "1/num_data", # fixes lambda
                      regularizor_parameter = "normalized",
                    #   regularizor_parameter = "Lbar/n",
                    #   repeat_stepsize_calculation=true, # used in minimizeFunc_grid_stepsize
                      initial_point="zeros", # is fixed not to add more randomness 
                      force_continue=false); # force continue if diverging or if tolerance reached

prob = load_logistic_from_matrices(X, y, data, options, lambda=-1, scaling="none");  # scaling = centering and scaling
println("Occupied memory: ", (Sys.total_memory() - Sys.free_memory()) / 2^20, " MiB")

varinfo()

X = nothing;
# y = nothing;
println("Occupied memory: ", (Sys.total_memory() - Sys.free_memory()) / 2^20, " MiB")

GC.gc()
println("Occupied memory: ", (Sys.total_memory() - Sys.free_memory()) / 2^20, " MiB")


### Trying to run BFGS to compute prob.fsol
options = set_options(tol=10.0^(-16.0), skip_error_calculation=20, exacterror=false, max_iter=10^8, 
        max_time=60.0*60.0*3.0, max_epocs=500, repeat_stepsize_calculation=true, rep_number=2);
options.batchsize = 1;
method_name = "BFGS";
# output = minimizeFunc_grid_stepsize(prob, method_name, options);
output = minimizeFunc(prob, "BFGS", options);
