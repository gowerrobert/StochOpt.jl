### EXPERIMENT 4

## Testing the optimality of our tau* for the same gamma = gamma_heuristic

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
using LaTeXStrings

default_path = "./data/";

## Bash inputs
# include("../src/StochOpt.jl") # Be carefull about the path here
# data = ARGS[1];
# scaling = ARGS[2];
# lambda = parse(Float64, ARGS[3]);
# println("Inputs: ", data, " + ", scaling, " + ",  lambda, "\n");

## Manual inputs
include("./src/StochOpt.jl") # Be carefull about the path here
datasets = readlines("$(default_path)available_datasets.txt");
idx = 8;
data = datasets[idx];
# scaling = "none";
scaling = "column-scaling";
lambda = 10^(-1);
# lambda = 10^(-3);


Random.seed!(1);

### LOADING THE DATA ###
println("--- Loading data ---");
X, y = loadDataset(default_path, data);

######################################## SETTING UP THE PROBLEM ########################################
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-4), max_iter=10^8, max_epocs=10^8,
                      max_time=60.0*60.0,
                      skip_error_calculation=10^4,
                      batchsize=1,
                      regularizor_parameter = "normalized",
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
    prob = load_ridge_regression(X, y, data, options, lambda=lambda, scaling=scaling); #column-scaling
end

n = prob.numdata;
d = prob.numfeatures;
mu = prob.mu;
Lmax = prob.Lmax;
L = prob.L;
Lbar = prob.Lbar;

if occursin("lgstc", prob.name)
    println("Correcting smoothness constants for logistic since phi'' <= 1/4")
    ## Correcting for logistic since phi'' <= 1/4 #TOCHANGE
    Lmax /= 4;
end

tau_heuristic = round(Int, 1 + (mu*(n-1))/(4*L))

filename = "lgstc_covtype_binary-column-scaling-regularizor-1e-01-exp4-optimality-1-avg-attempt4";
minibatchlist, itercomplex, empcomplex, tau_empirical = load("$(default_path)$(filename).jld", "minibatchlist", "itercomplex", "empcomplex", "tau_empirical")

pyplot()
plot_empirical_complexity(prob, minibatchlist, empcomplex, tau_heuristic, tau_empirical);
