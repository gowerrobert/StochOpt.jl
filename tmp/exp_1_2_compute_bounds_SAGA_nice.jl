### EXPERIMENT 1 & 2

## Computing the upper-bounds of the expected smoothness constant (exp. 1)
## and our step sizes (exp. 2)

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
include("../src/StochOpt.jl") # Be carefull about the path here
default_path = "./data/";
data = ARGS[1];
scaling = ARGS[2];
lambda = parse(Float64, ARGS[3]);
println("Inputs: ", data, " + ", scaling, " + ",  lambda, "\n");

## Manual inputs
# include("./src/StochOpt.jl") # Be carefull about the path here
# default_path = "./data/";
# datasets = readlines("$(default_path)available_datasets.txt");
# idx = 7; # YearPredictionMSD
# data = datasets[idx];
# scaling = "none";
# # scaling = "column-scaling";
# # # lambda = -1;
# lambda = 10^(-3);
# # lambda = 10^(-1);

Random.seed!(1);

### LOADING DATA ###
println("--- Loading data ---");
## Only loading datasets, no data generation
X, y = loadDataset(default_path, data);

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-1), max_iter=10^8, max_time=10.0^2, max_epocs=10^8,
                    #   regularizor_parameter = "1/num_data", # fixes lambda
                      regularizor_parameter = "normalized",
                    #   regularizor_parameter = "Lbar/n",
                    #   repeat_stepsize_calculation=true, # used in minimizeFunc_grid_stepsize
                      initial_point="zeros", # is fixed not to add more randomness
                      force_continue=false); # force continue if diverging or if tolerance reached
u = unique(y);
if length(u) < 2
    error("Wrong number of possible outputs")
elseif length(u) == 2
    println("Binary output detected: the problem is set to logistic regression")
    prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling=scaling);
else
    println("More than three modalities in the outputs: the problem is set to ridge regression")
    prob = load_ridge_regression(X, y, data, options, lambda=lambda, scaling=scaling);
end

n = prob.numdata;
d = prob.numfeatures;

### COMPUTING THE SMOOTHNESS CONSTANTS ###
# Compute the smoothness constants L, L_max, \cL, \bar{L}
datathreshold = 24; # if n is too large we do not compute the exact expected smoothness constant nor its relative quantities

expsmoothcst = nothing;

########################### EMPIRICAL UPPER BOUNDS OF THE EXPECTED SMOOTHNESS CONSTANT ###########################
#region
### COMPUTING THE BOUNDS ###
simplebound, bernsteinbound, heuristicbound, expsmoothcst = get_expected_smoothness_bounds(prob); # WARNING : markers are missing!

### PLOTING ###
println("\n--- Ploting upper bounds ---");
# PROBLEM: there is still a problem of ticking non integer on the xaxis
pyplot()
plot_expected_smoothness_bounds(prob, simplebound, bernsteinbound, heuristicbound, expsmoothcst);

# heuristic equals true expected smoothness constant for tau=1 and n as expected, else it is above as hoped
if(n<=datathreshold)
    println("Heuristic - expected smoothness gap: ", heuristicbound - expsmoothcst)
    println("Simple - heuristic gap: ", simplebound - heuristicbound)
    println("Bernstein - simple gap: ", bernsteinbound - simplebound)
end
#endregion
##################################################################################################################


##################################### EMPIRICAL UPPER BOUNDS OF THE STEPSIZES ####################################
#region
## TO BE DONE: implement grid-search for the stepsizes, i.e.
## 1) set a grid of stepsizes around 1/(4Lmax)
## 2) run several SAGA_nice on the same problem with different stepsize (average?)
## 3) pick the 'best' stepsize

### COMPUTING THE UPPER-BOUNDS OF THE STEPSIZES ###
simplestepsize, bernsteinstepsize, heuristicstepsize, hofmannstepsize, expsmoothstepsize = get_stepsize_bounds(prob, simplebound, bernsteinbound, heuristicbound, expsmoothcst);

### PLOTING ###
println("\n--- Ploting stepsizes ---");
# PROBLEM: there is still a problem of ticking non integer on the xaxis
pyplot()
plot_stepsize_bounds(prob, simplestepsize, bernsteinstepsize, heuristicstepsize, hofmannstepsize, expsmoothstepsize);
#endregion
##################################################################################################################


###################################### THEORETICAL OPTIMAL MINI-BATCH SIZES ######################################
#region
## Compute optimal mini-batch size
if typeof(expsmoothcst)==Array{Float64,2}
    LHS = 4*(1:n).*(expsmoothcst .+ prob.lambda)./prob.mu;
    RHS = n .+ (n .- (1:n)) .* (4*(prob.Lmax+prob.lambda)/((n-1)*prob.mu));
    exacttotalcplx = reshape(max.(LHS, RHS), n);
    opt_minibatch_exact = argmin(exacttotalcplx);
else
    opt_minibatch_exact = nothing;
end

## WARNING: Verify computations : should we add lambda????
opt_minibatch_simple = round(Int, 1 + (prob.mu*(n-1))/(4*prob.Lbar)); # One should not add again lambda since it is already taken into account in Lbar
opt_minibatch_bernstein = max(1, round(Int, 1 + (prob.mu*(n-1))/(8*prob.L) - (4/3)*log(d)*((n-1)/n)*(prob.Lmax/(2*prob.L)) )); ## WARNING: Verify computations : should we add lambda????
opt_minibatch_heuristic = round(Int, 1 + (prob.mu*(n-1))/(4*prob.L));
#endregion
##################################################################################################################


########################################### SAVNG RESULTS ########################################################
#region
save_SAGA_nice_constants(prob, data, simplebound, bernsteinbound, heuristicbound, expsmoothcst,
                         simplestepsize, bernsteinstepsize, heuristicstepsize, expsmoothstepsize,
                         opt_minibatch_simple, opt_minibatch_bernstein, opt_minibatch_heuristic,
                         opt_minibatch_exact);
#endregion
##################################################################################################################
