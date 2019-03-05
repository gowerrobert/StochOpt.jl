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

## Bash inputs
include("../src/StochOpt.jl") # Be carefull about the path here
default_path = "./data/";
data = ARGS[1];
scaling = ARGS[2];
lambda = parse(Float64, ARGS[3]);
stepsize_multiplier = parse(Float64, ARGS[4]);
println("Inputs: ", data, " + ", scaling, " + ", lambda, " + stepsize_multiplier = ",  stepsize_multiplier, "\n");

## Manual inputs
# include("./src/StochOpt.jl") # Be carefull about the path here
# default_path = "./data/";
# datasets = readlines("$(default_path)available_datasets.txt");
# idx = 11;
# data = datasets[idx];
# scaling = "none";
# # scaling = "column-scaling";
# # lambda = -1;
# lambda = 10^(-3);
# # lambda = 10^(-1);
# stepsize_multiplier = 2^(-1.0);

Random.seed!(1);

### LOADING DATA ###
println("--- Loading data ---");
@time X, y = loadDataset(default_path, data);

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-16.0), skip_error_calculation=1,
                      exacterror=false, max_iter=10^8,
                      max_time=60.0*60.0, max_epocs=2000, force_continue=true);
@time prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling=scaling);


########################################### covtype.binary ############################################
#region
## Computing the solution with a serial gridsearch
# @time get_fsol_logistic!(prob)

options = set_options(tol=10.0^(-16.0), skip_error_calculation=1, exacterror=false, max_iter=10^8,
                              max_time=60.0, max_epocs=10^5, repeat_stepsize_calculation=true, rep_number=2);
## Running BFGS
options.batchsize = prob.numdata;
method_input = "BFGS";
# grid = [2.0^(9), 2.0^(7), 2.0^(5), 2.0^(3), 2.0^(1), 2.0^(-1), 2.0^(-3), 2.0^(-5)];
# output = minimizeFunc_grid_stepsize(prob, method_input, options, grid=grid);
output = minimizeFunc_grid_stepsize(prob, method_input, options);

## Result = stepsize = 0.5 for none and both lambdas
## Result = stepsize =  for column-scaling

## Running BFGS
options = set_options(tol=10.0^(-16.0), skip_error_calculation=5,
                      exacterror=false, max_iter=10^8,
                      max_time=60.0*60.0,
                      max_epocs=50,
                      force_continue=true);
options.stepsize_multiplier = stepsize_multiplier;
options.batchsize = prob.numdata;
method_input = "BFGS";
output = minimizeFunc(prob, method_input, options);

## Saving the optimization output in a JLD file
a, savename = get_saved_stepsize(prob.name, method_input, options);
a = nothing;
save("$(default_path)$(savename).jld", "output", output)

## Setting the true solution as the smallest of both
prob.fsol = minimum(output.fs[.!isnan.(output.fs)]);
println("\n----------------------------------------------------------------------")
@printf "For %s, fsol = %f\n" prob.name prob.fsol
println("----------------------------------------------------------------------\n")

## Saving the solution in a JLD file
fsolfilename = get_fsol_filename(prob); # not coherent with get_saved_stepsize output
save("$(fsolfilename).jld", "fsol", prob.fsol);
#endregion
######################################################################################################
