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
# include("../src/StochOpt.jl") # Be carefull about the path here
# default_path = "./data/";
# data = ARGS[1];
# scaling = ARGS[2];
# lambda = parse(Float64, ARGS[3]);
# stepsize_multiplier = parse(Float64, ARGS[4]);
# println("Inputs: ", data, " + ", scaling, " + ", lambda, " + stepsize_multiplier = ",  stepsize_multiplier, "\n");

## Manual inputs
include("./src/StochOpt.jl") # Be carefull about the path here
default_path = "./data/";
# datasets = readlines("$(default_path)available_datasets.txt");
# idx = 11;
# data = datasets[idx];
data = "real-sim"
scaling = "none";
# scaling = "column-scaling";
# lambda = -1;
lambda = 10^(-3);
# lambda = 10^(-1);
stepsize_multiplier = 2^(-1.0);

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

## Result = stepsize = 0.125 for none and lambda = 0.001

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

n = prob.numdata;
d = prob.numfeatures;
mu = prob.mu
Lmax = prob.Lmax;
L = prob.L;
# Lbar = prob.Lbar;

options = set_options(max_iter=10^8, max_time=10.0^(5.0), max_epocs=8, force_continue=true, initial_point="zeros",
                      exacterror=false);

## First test of SAGA-nice (hand-made settings)
options.stepsize_multiplier = 1e-2;
options.batchsize = 1;
options.skip_error_calculation = 5000;
SAGA_nice = initiate_SAGA_nice(prob, options); # separated implementation from SAGA
output = minimizeFunc(prob, SAGA_nice, options);

## Test with theoretical parameters
b_practical = round(Int, 1 + (mu*(n-1))/(4*L))
# rho = ( n*(n - b_practical) ) / ( b_practical*(n-1) ); # Sketch residual rho = n*(n-b)/(b*(n-1)) in JacSketch paper, page 35
rightterm = ( Lmax*(n - b_practical) ) / ( b_practical*(n-1) ) + ( (mu*n) / (4*b_practical) ); # Right-hand side term in the max
practical_bound = ( n*(b_practical-1)*L + (n-b_practical)*Lmax ) / ( b_practical*(n-1) );
step_practical = 0.25 / max(practical_bound, rightterm);

options.stepsize_multiplier = step_practical
options.batchsize = b_practical
options.skip_error_calculation = 1;
SAGA_nice = initiate_SAGA_nice(prob, options); # separated implementation from SAGA
output = minimizeFunc(prob, SAGA_nice, options);


## Test in relative error
@time X, y = loadDataset(default_path, data);

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(max_iter=10^8, max_time=10.0^(5.0), max_epocs=8, force_continue=true, initial_point="zeros");
@time prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling=scaling);

options.stepsize_multiplier = step_practical
options.batchsize = b_practical
options.skip_error_calculation = 1;
SAGA_nice = initiate_SAGA_nice(prob, options); # separated implementation from SAGA
output = minimizeFunc(prob, SAGA_nice, options);