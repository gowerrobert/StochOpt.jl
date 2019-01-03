## EXPERIMENT 3

## Comparing different classical settings of SAGA and ours
# using Distributed

# addprocs(4)

# @everywhere begin # this part will be available on all CPUs
#     using JLD
#     using Plots
#     using StatsBase
#     using Match
#     using Combinatorics
#     using Random
#     using Printf
#     using LinearAlgebra
#     using Statistics
#     using Base64
#     using SharedArrays
#     include("./src/StochOpt.jl") # Be carefull about the path here
# end

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

default_path = "./data/";

println("--- Loading data ---");
datasets = readlines("$(default_path)available_datasets.txt");
# idx = 4; # australian
idx = 3; # YearPredictionMSD
data = datasets[idx];
X, y = loadDataset(data);

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
scaling = "none";
# scaling = "column-scaling";
options = set_options(tol=10.0^(-10), max_iter=10^8, max_time=5.0, max_epocs=10^8, 
                      skip_error_calculation=10^3, batchsize=1, 
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

if occursin("lgstc", prob.name) # julia 0.7
    ## Correcting for logistic since phi'' <= 1/4
    Lmax /= 4;
end

### I) tau = 1 ###

##---------- Computing step sizes ----------
step_defazio = 1.0 / (3.0*(Lmax + n*mu))
K = (4.0*Lmax) / (n*mu);
step_hofmann = K / (2*Lmax*(1+K+sqrt(1+K^2)))
step_heuristic = 1.0 / (4.0*Lmax + n*mu)

## Calculating best stepsize for SAGA_nice on ridge_housing-none with batchsize 1
options = set_options(tol=10.0^(-1), max_iter=10^8, max_time=0.1, max_epocs=10^8, 
                      skip_error_calculation=100, batchsize=1, rep_number=2, 
                      force_continue=false, repeat_stepsize_calculation = true,
                      regularizor_parameter="normalized", initial_point="zeros");
SAGA_nice = initiate_SAGA_nice(prob, options);
grid = [2.0^(0), 2.0^(-29)];
minimizeFunc_grid_stepsize(prob, SAGA_nice, options, grid=grid);
# parallel_minimizeFunc_grid_stepsize(prob, SAGA_nice, options, grid=grid)


function calculate_best_stepsize_SAGA_nice(prob, options ; skip, max_time, rep_number, grid)
    options.repeat_stepsize_calculation = true;
    options.rep_number = rep_number;
    options.skip_error_calculation = skip;
    options.tol = 10.0^(-1);
    options.max_iter = 10^8;
    options.max_epocs = 10^5;
    options.max_time = max_time;
    SAGA_nice = initiate_SAGA_nice(prob, options);
    output = minimizeFunc_grid_stepsize(prob, SAGA_nice, options, grid=grid)
    # output = parallel_minimizeFunc_grid_stepsize(prob, SAGA_nice, options, grid=grid)
    return output
end

# Warning SAGA-nice too look for step size but method is called SAGA_nice
step_gridsearch, = get_saved_stepsize(prob.name, "SAGA-nice", options)
if step_gridsearch == 0.0
    # grid = [2.0^(-21), 2.0^(-23), 2.0^(-25), 2.0^(-27), 2.0^(-29), 2.0^(-31), 2.0^(-33)];
    # grid = [2.0^(-31), 2.0^(-33)];
    # include("./src/StochOpt.jl") # Be carefull about the path here
    grid = [2.0^(-7), 2.0^(-35)];
    Random.seed!(1);
    output1 = calculate_best_stepsize_SAGA_nice(prob, options, skip=10, max_time=0.0025, rep_number=1, grid=grid);

    step_gridsearch = get_saved_stepsize(prob.name, "SAGA-nice", options);
end


















stepsizes_names = ["Defazio_et_al", "Hofmann_et_al", "Heuristic", "Grid_search"];
stepsizes = [step_defazio, step_hofmann, step_heuristic, step_gridsearch]


##---------- SAGA_nice-1 runs ----------
options = set_options(tol=10.0^(-10), max_iter=10^8, max_time=5.0, max_epocs=10^8, 
                      skip_error_calculation=10^3, batchsize=1, 
                    #   regularizor_parameter = "1/num_data", # fixes lambda
                      regularizor_parameter = "normalized",
                    #   regularizor_parameter = "Lbar/n",
                      initial_point="zeros", # is fixed not to add more randomness 
                      force_continue=false); # force continue if diverging or if tolerance reached
options.force_continue = true;
numsimu = 1;
skip_multiplier = 0.02;

probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
itercomplex = zeros(length(stepsizes), 1);
OUTPUTS = [];
for idxstep in 1:length(stepsizes)
    println("\nCurrent step size: ", stepsizes_names[idxstep], " = ", stepsizes[idxstep]);
        
    options.stepsize_multiplier = stepsizes[idxstep];
    for i=1:numsimu
        println("----- Simulation #", i, " -----");
        sg = initiate_SAGA_nice(prob, options); # separated implementation from SAGA
        println("---> Step size: ",  stepsizes_names[idxstep], " = ", @sprintf "%0.2e" sg.stepsize);
        output = minimizeFunc(prob, sg, options, stop_at_tol=true);
        println("---> Output fail = ", output.fail, "\n");
        itercomplex[idxstep] += output.iterations;
        output.name = string(stepsizes_names[idxstep], " = ", @sprintf "%0.2e" stepsizes[idxstep]);
        global OUTPUTS = [OUTPUTS; output];
    end
end
itercomplex = itercomplex ./ numsimu; # simply averaging the last iteration number
itercomplex = itercomplex[:];

## Saving the result of the simulations
# savename = "-empcomplex-$(numsimu)-avg";
# save("$(default_path)$(probname)$(savename).jld", "itercomplex", itercomplex, "OUTPUTS", OUTPUTS);


## Checking that all simulations reached tolerance
fails = [OUTPUTS[i].fail for i=1:length(stepsizes)*numsimu];
if all(s->(string(s)=="tol-reached"), fails)
    println("Tolerance always reached")
end

## Plotting one SAGA-nice simulation for each mini-batch size
if(numsimu==1)
    gr()
    # pyplot()
    plot_outputs_Plots(OUTPUTS, prob, options); # Plot and save output
end



### II) tau = tau* ###
# Hofmann : tau = 20, gamma = gamma(20)