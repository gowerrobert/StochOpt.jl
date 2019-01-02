## EXPERIMENT 3

## Comparing different classical settings of SAGA and ours

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

include("./src/StochOpt.jl") # Be carefull about the path here

include("./src/get_saved_stepsize.jl")

default_path = "./data/";

Random.seed!(1);

println("--- Loading data ---");
datasets = readlines("$(default_path)available_datasets.txt");
idx = 4; # australian
data = datasets[idx];
X, y = loadDataset(data);

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
scaling = "none";
# scaling = "column-scaling";
options = set_options(tol=10.0^(-1), max_iter=10^8, max_time=5.0, max_epocs=10^8, 
                      skip_error_calculation=10^2, batchsize=1, 
                    #   regularizor_parameter = "1/num_data", # fixes lambda
                      regularizor_parameter = "normalized",
                    #   regularizor_parameter = "Lbar/n",
                      initial_point="zeros", # is fixed not to add more randomness 
                      force_continue=false); # force continue if diverging or if tolerance reached
u = unique(y);
if length(u) < 2
    error("Wrong number of possible outputs")
elseif length(u) == 2
    println("Binary output detected: the problem is set to logistic regression")
    prob = load_logistic_from_matrices(X, y, data, options, lambda=-1, scaling=scaling);  
else
    println("More than three modalities in the outputs: the problem is set to ridge regression")
    prob = load_ridge_regression(X, y, data, options, lambda=-1, scaling=scaling); #column-scaling
end

n = prob.numdata;
d = prob.numfeatures;
mu = prob.mu;
Lmax = prob.Lmax;

### I) tau = 1 ###
step_defazio = 1.0 / (3.0*(Lmax + n*mu))
K = (4.0*Lmax) / (n*mu);
step_hofmann = K / (2*Lmax*(1+K+sqrt(1+K^2)))
step_heuristic = 1.0 / (4.0*Lmax + n*mu)


## Calculating best stepsize for SAGA_nice on ridge_housing-none with batchsize 1
function calculate_best_stepsize_SAGA_nice(prob, options)
    options.repeat_stepsize_calculation = true;
    options.skip_error_calculation = 10^5;
    options.tolerance = 10.0^(-16);
    options.max_iter = 10^8;
    options.max_epocs = 10^5;
    options.max_time = 60.0*60.0*3.0;
    SAGA_nice = initiate_SAGA_nice(prob, options);
    output1 = minimizeFunc_grid_stepsize(prob, SAGA_nice, options)
end


SAGA_nice = initiate_SAGA_nice(prob, options); # new separated implementation
options.repeat_stepsize_calculation = true;
options.skip_error_calculation = 10^3;
output1 = minimizeFunc_grid_stepsize(prob, SAGA_nice, options)
# get_fsol_logistic!(prob)



step_gridsearch = get_saved_stepsize(prob.name, "SAGA-nice", options) # Warning SAGA-nice too look for step size but method is called SAGA_nice
# lgstc_australian-column-scaling-stepsizes.txt




stepsizes = [step_defazio; step_hofmann; step_opt; step_gridsearch]


### II) tau = tau* ###
# Hofmann : tau = 20, gamma = gamma(20)