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
# data = "a1a_full"
# data = "colon-cancer"
data = "leukemia_full"
@time X, y = loadDataset(default_path, data)

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-1), max_iter=10^8, max_time=10.0^2, max_epocs=10^8,
                    #   regularizor_parameter = "1/num_data", # fixes lambda
                      regularizor_parameter = "normalized",
                    #   regularizor_parameter = "Lbar/n",
                      initial_point="zeros", # is fixed not to add more randomness
                      force_continue=false); # force continue if diverging or if tolerance reached

@time prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-1, scaling="none");
# @time prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-3, scaling="none");
prob.fsol

##########################################################################################
#region
## Computing the solution with a serial gridsearch
@time get_fsol_logistic!(prob);


options = set_options(tol=10.0^(-16.0), skip_error_calculation=100, exacterror=false, max_iter=10^8,
                      max_time=60.0*30.0, max_epocs=10^7, force_continue=true);
## Running BFGS
# options.stepsize_multiplier = 128; # colon-cancer 1e-3
# options.stepsize_multiplier = 512; # colon-cancer 1e-1
options.stepsize_multiplier = 2048; # leukemia_full 1e-1
# options.stepsize_multiplier = 128; # leukemia_full 1e-3

options.batchsize = prob.numdata;
method_input = "BFGS";
output = minimizeFunc(prob, method_input, options);

## Saving the optimization output in a JLD file
a, savename = get_saved_stepsize(prob.name, method_input, options);
a = nothing;
save("$(default_path)$(savename).jld", "output", output)

## Setting the true solution as the smallest of both
prob.fsol = minimum(output.fs);
println("\n----------------------------------------------------------------------")
@printf "For %s, fsol = %f\n" prob.name prob.fsol
println("----------------------------------------------------------------------\n")

## Saving the solution in a JLD file
fsolfilename = get_fsol_filename(prob); # not coherent with get_saved_stepsize output
save("$(fsolfilename)_try_2.jld", "fsol", prob.fsol)
#endregion
######################################################################################################
