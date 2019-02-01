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
datasets = readlines("$(default_path)available_datasets.txt");
idx = 8;
data = datasets[idx];
scaling = "none";
# scaling = "column-scaling";
# lambda = -1;
# lambda = 10^(-3);
lambda = 10^(-1);
stepsize_multiplier = 2^(-5.0);

Random.seed!(1);

### LOADING DATA ###
println("--- Loading data ---");
@time X, y = loadDataset(default_path, data);

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-16.0), skip_error_calculation=100,
                      exacterror=false, max_iter=10^8,
                      max_time=60.0*60.0, max_epocs=100, force_continue=true);
@time prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling=scaling);

X = nothing;
y = nothing;

########################################### rcv1_full ############################################
#region
## Computing the solution with a serial gridsearch
# @time get_fsol_logistic!(prob)

## Running SVRG
method_input = "SVRG";
options.stepsize_multiplier = stepsize_multiplier;
options.batchsize = 100;
options.skip_error_calculation=100;
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

save("$(default_path)prob_backup_$(prob.name).jld",
     "name", prob.name, "numdata", prob.numdata, "numfeatures", prob.numfeatures,
     "X", prob.X, "y", prob.y,
     "L", prob.L, "Lbar", prob.Lbar, "Lmax", prob.Lmax,
     "mu", prob.mu, "lmabda", prob.lambda);

## Saving the solution in a JLD file
fsolfilename = get_fsol_filename(prob);
save("$(fsolfilename).jld", "fsol", prob.fsol);
#endregion
######################################################################################################
