### EXPERIMENT 3

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
using Formatting

## Bash inputs
# include("../src/StochOpt.jl") # Be carefull about the path here
# data = ARGS[1];
# scaling = ARGS[2];
# lambda = parse(Float64, ARGS[3]);
# println("Inputs: ", data, " + ", scaling, " + ",  lambda, "\n");

## Manual inputs
include("./src/StochOpt.jl") # Be carefull about the path here
default_path = "./data/";
datasets = readlines("$(default_path)available_datasets.txt");
idx = 3; # YearPredictionMSD
data = datasets[idx];
# scaling = "none";
scaling = "column-scaling";
# lambda = -1;
lambda = 10^(-3);

Random.seed!(1);

### LOADING THE DATA ###
println("--- Loading data ---");
default_path = "./data/";
datasets = readlines("$(default_path)available_datasets.txt");
# idx = 4; # australian
idx = 3; # YearPredictionMSD
data = datasets[idx];
X, y = loadDataset(default_path, data);

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-6), max_iter=10^8, max_epocs=10^8,
                      max_time=120.0,
                      skip_error_calculation=10^5,
                      batchsize=1,
                      regularizor_parameter = "normalized",
                      initial_point="zeros", # is fixed not to add more randomness
                      force_continue=false); # force continue if diverging or if tolerance reached
u = unique(y);
if length(u) < 2
    error("Wrong number of possible outputs");
elseif length(u) == 2
    println("Binary output detected: the problem is set to logistic regression")
    prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling=scaling);
else
    println("More than three modalities in the outputs: the problem is set to ridge regression")
    prob = load_ridge_regression(X, y, data, options, lambda=lambda, scaling=scaling); #column-scaling
end

n = prob.numdata;
d = prob.numfeatures;
mu = prob.mu
Lmax = prob.Lmax;
L = prob.L;

if occursin("lgstc", prob.name) # julia 0.7
    ## Correcting for logistic since phi'' <= 1/4
    Lmax /= 4;
end

### I) tau = 1 ###
#region
##---------- Computing step sizes ----------
step_defazio = 1.0 / (3.0*(Lmax + n*mu))
K = (4.0*Lmax) / (n*mu);
step_hofmann = K / (2*Lmax*(1+K+sqrt(1+K^2)));
# step_hofmann = 1.0 / (mu*n); # ridiculously large if mu is very small
step_heuristic = 1.0 / (4.0*Lmax + n*mu);

## Calculating best grid search step size for SAGA_nice with batchsize 1
function calculate_best_stepsize_SAGA_nice(prob, options ; skip, max_time, rep_number, batchsize, grid)
    old_skip = options.skip_error_calculation;
    old_tol = options.tol;
    old_max_iter = options.max_iter;
    old_max_epocs = options.max_epocs;
    old_max_time = options.max_time;
    old_rep_number = options.rep_number;
    old_batchsize = options.batchsize;

    options.repeat_stepsize_calculation = true;
    options.rep_number = rep_number;
    options.skip_error_calculation = skip;
    options.tol = 10.0^(-6);
    options.max_iter = 10^8;
    options.max_epocs = 10^5;
    options.max_time = max_time;
    options.batchsize = batchsize;
    SAGA_nice = initiate_SAGA_nice(prob, options);
    output = minimizeFunc_grid_stepsize(prob, SAGA_nice, options, grid=grid);

    options.repeat_stepsize_calculation = false;
    options.skip_error_calculation = old_skip;
    options.tol = old_tol;
    options.max_iter = old_max_iter;
    options.max_epocs = old_max_epocs;
    options.max_time = old_max_time;
    options.rep_number = old_rep_number;
    options.batchsize = old_batchsize;
    return output
end

# Warning SAGA-nice too look for step size but method is called SAGA_nice
# step_gridsearch, = get_saved_stepsize(prob.name, "SAGA-nice", options)
# if step_gridsearch == 0.0
    grid = [2.0^(25), 2.0^(23), 2.0^(21), 2.0^(19), 2.0^(17), 2.0^(15), 2.0^(13), 2.0^(11),
            2.0^(9), 2.0^(7), 2.0^(5), 2.0^(3), 2.0^(1), 2.0^(-1), 2.0^(-3), 2.0^(-5),
            2.0^(-7), 2.0^(-9), 2.0^(-11), 2.0^(-13), 2.0^(-15), 2.0^(-17), 2.0^(-19),
            2.0^(-21), 2.0^(-23), 2.0^(-25), 2.0^(-27), 2.0^(-29), 2.0^(-31), 2.0^(-33)];
    output = calculate_best_stepsize_SAGA_nice(prob, options, skip=10^5, max_time=10.0,
                                               rep_number=3, batchsize=1, grid=grid);

    step_gridsearch, = get_saved_stepsize(prob.name, "SAGA-nice", options);
# end

# method_names = ["Grid_search", "Defazio_et_al", "Hofmann_et_al", "Heuristic"];
method_names = ["SAGA + grid search", "SAGA", "q-SAGA (q=1)", "SAGA heuristic"];
stepsizes = [step_gridsearch, step_defazio, step_hofmann, step_heuristic];

##---------- SAGA_nice-1 runs ----------
# options = set_options(tol=10.0^(-6), max_iter=10^8, max_epocs=10^8,
#                       max_time=120.0,
#                       skip_error_calculation=10^4,
#                       batchsize=1,
#                       regularizor_parameter = "normalized",
#                       initial_point="zeros", # is fixed not to add more randomness
#                       force_continue=false); # force continue if diverging or if tolerance reached

function calculate_skip_error(stepsize)
    if 1e-8 <= stepsize
        skip = 100;
    elseif 1e-9 <= stepsize < 1e-8
        skip = 200;
    elseif 1e-10 <= stepsize < 1e-9
        skip = 500;
    elseif 5e-11 <= stepsize < 1e-10
        skip = 1000;
    elseif 1e-11 <= stepsize < 5e-11
        skip = 10000;
    else
        skip = 10000;
    end
end
# skip_error = calculate_skip_error.(stepsizes);
# skip_error = [10^2, 10^3, 10^3, 10^3];
skip_error = [10^5, 10^5, 10^5, 10^5];
numsimu = 1;
itercomplex = zeros(length(stepsizes), 1);
OUTPUTS = [];
for idxstep in 1:length(stepsizes)
    options.stepsize_multiplier = stepsizes[idxstep];
    for i=1:numsimu
        println("\n----- Simulation #", i, " -----");
        options.skip_error_calculation = skip_error[idxstep]; # compute a skip error for each step size
        SAGA_nice = initiate_SAGA_nice(prob, options); # separated implementation from SAGA
        println("Current step size: ", method_names[idxstep], " = ", stepsizes[idxstep]);
        output = minimizeFunc(prob, SAGA_nice, options, stop_at_tol=true);
        println("---> Output fail = ", output.fail, "\n");
        itercomplex[idxstep] += output.iterations;
        output.name = string(method_names[idxstep]);
        global OUTPUTS = [OUTPUTS; output];
    end
end
itercomplex = itercomplex ./ numsimu; # simply averaging the last iteration number
itercomplex = itercomplex[:];

## Saving the result of the simulations
probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
savename = string(probname, "-exp3_1-empcomplex-", numsimu, "-avg");
save("$(default_path)$(savename).jld", "itercomplex", itercomplex, "OUTPUTS", OUTPUTS,
     "method_names", method_names,"stepsizes", stepsizes);

## Checking that all simulations reached tolerance
fails = [OUTPUTS[i].fail for i=1:length(stepsizes)*numsimu];
if all(s->(string(s)=="tol-reached"), fails)
    println("Tolerance always reached")
end

## Plotting one SAGA-nice simulation for each mini-batch size
if numsimu == 1
    gr()
    # pyplot()
    plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp3_1"); # Plot and save output
end

@printf "\n|  %s  | %s | %s |  %s   |\n" method_names[1] method_names[2] method_names[3] method_names[4]
@printf "| %e  | %e  | %e  | %e |\n\n" stepsizes[1] stepsizes[2] stepsizes[3] stepsizes[4]
@printf "| %d  | %d  | %d  | %d |\n\n" itercomplex[1] itercomplex[2] itercomplex[3] itercomplex[4]

#endregion

### II) tau = tau* ###
# Hofmann : tau = 20, gamma = gamma(20)
##---------- Computing step sizes ----------
tau_defazio = 1;
step_defazio = 1.0 / (3.0*(Lmax + n*mu));

tau_hofmann = 20;
K = (4.0*tau_hofmann*Lmax) / (n*mu);
step_hofmann = K / (2*Lmax*(1+K+sqrt(1+K^2)));
# step_hofmann = tau/(mu*n);

rho = ( n*(n - tau_hofmann) ) / ( tau_hofmann*(n-1) ); # Sketch residual
rightterm = (rho / n)*Lmax + ( (mu*n) / (4*tau_hofmann) ); # Right-hand side term in the max
heuristicbound = ( n*(tau_hofmann-1)*L + (n-tau_hofmann)*Lmax ) / ( tau_hofmann*(n-1) );
step_hofmann_heuristic = 0.25 / max(heuristicbound, rightterm);

## IS our optimal tau always one???
## YearPredictionMSD scaled + mu = 10^(-3) => 13
## YearPredictionMSD scaled + mu = 10^(-1) => 1245
tau_heuristic = round(Int, 1 + ( mu*(n-1) ) / ( 4*L ) );
# tau_heuristic = 20;
rho = ( n*(n - tau_heuristic) ) / ( tau_heuristic*(n-1) ); # Sketch residual
rightterm = (rho / n)*Lmax + ( (mu*n) / (4*tau_heuristic) ); # Right-hand side term in the max
heuristicbound = ( n*(tau_heuristic-1)*L + (n-tau_heuristic)*Lmax ) / ( tau_heuristic*(n-1) );
step_heuristic = 0.25 / max(heuristicbound, rightterm);

## Calculating best grid search step size for SAGA_nice with batchsize >= 1
options = set_options(tol=10.0^(-6), 
                      max_iter=10^8, 
                      max_epocs=10^8, 
                      max_time=120.0, 
                      skip_error_calculation=10^5,
                      regularizor_parameter = "normalized", 
                      initial_point="zeros", 
                      force_continue=false,
                      batchsize=tau_heuristic);
if options.batchsize == 1
    method_name = "SAGA-nice";
elseif options.batchsize > 1
    method_name = string("SAGA-", options.batchsize, "-nice");
else
    error("Invalid batch size");
end
# step_heuristic_gridsearch, = get_saved_stepsize(prob.name, method_name, options);
# if step_heuristic_gridsearch == 0.0
    grid = [2.0^(25), 2.0^(23), 2.0^(21), 2.0^(19), 2.0^(17), 2.0^(15), 2.0^(13), 2.0^(11),
            2.0^(9), 2.0^(7), 2.0^(5), 2.0^(3), 2.0^(1), 2.0^(-1), 2.0^(-3), 2.0^(-5),
            2.0^(-7), 2.0^(-9), 2.0^(-11), 2.0^(-13), 2.0^(-15), 2.0^(-17), 2.0^(-19),
            2.0^(-21), 2.0^(-23), 2.0^(-25), 2.0^(-27), 2.0^(-29), 2.0^(-31), 2.0^(-33)];
    output = calculate_best_stepsize_SAGA_nice(prob, options, skip=10^1, max_time=120.0,
                                               rep_number=5, batchsize=tau_heuristic, grid=grid);
    step_heuristic_gridsearch, = get_saved_stepsize(prob.name, method_name, options);
# end

# method_names = ["heuristic + grid search", "SAGA", "q-SAGA", "heuristic"];
# mini_batch_sizes = [tau_heuristic, tau_defazio, tau_hofmann, tau_heuristic];
# stepsizes = [step_heuristic_gridsearch, step_defazio, step_hofmann, step_heuristic];
method_names = ["heuristic + grid search", "SAGA", "q-SAGA", "q-SAGA + heuristic", "heuristic"];
mini_batch_sizes = [tau_heuristic, tau_defazio, tau_hofmann, tau_hofmann, tau_heuristic];
stepsizes = [step_heuristic_gridsearch, step_defazio, step_hofmann, step_hofmann_heuristic, step_heuristic];

##---------- SAGA_nice-1 runs ----------
# options = set_options(tol=10.0^(-6), 
#                       max_time=120.0, 
#                       skip_error_calculation=10^3,                      
#                       max_iter=10^8, 
#                       max_epocs=10^8, 
#                       regularizor_parameter="normalized", initial_point="zeros", force_continue=true);
# skip_error = [10^2, 10^3, 10^3, 10^3];
# skip_error = [10^1, 10^4, 10^3, 10^3, 10^1]; # skip = n/(tau*10) approx 10 pass for 1 epoch
skip_error = closest_power_of_ten.(round.(Int, n ./ (10*mini_batch_sizes))) # 5 points per epoch
numsimu = 1;
array_val = SharedArray{Float64}(n, m);
    @sync @distributed for idx = 1:n*m
itercomplex = zeros(length(stepsizes), 1);
OUTPUTS = [];
for idxmethod in 1:length(stepsizes)
    options.stepsize_multiplier = stepsizes[idxmethod];
    for i=1:numsimu
        println("\n----- Simulation #", i, " -----");
        options.skip_error_calculation = skip_error[idxmethod]; # compute a skip error for each step size
        options.batchsize = mini_batch_sizes[idxmethod];
        SAGA_nice = initiate_SAGA_nice(prob, options);
        println("Current method: ", method_names[idxmethod], ", mini-batch size = ", mini_batch_sizes[idxmethod],
                ", step size = ", stepsizes[idxmethod]);
        output = minimizeFunc(prob, SAGA_nice, options, stop_at_tol=true);
        println("---> Output fail = ", output.fail, "\n");
        itercomplex[idxmethod] += output.iterations;
        # println("name = ", output.name)
        output.name = string(method_names[idxmethod]);
        global OUTPUTS = [OUTPUTS; output];
    end
end
itercomplex = itercomplex ./ numsimu; # simply averaging the last iteration number
itercomplex = itercomplex[:];
empcomplex = mini_batch_sizes .* itercomplex;

## Saving the result of the simulations
probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
savename = string(probname, "-exp3_2-empcomplex-", numsimu, "-avg");
save("$(default_path)$(savename).jld", "itercomplex", itercomplex, "OUTPUTS", OUTPUTS,
     "method_names", method_names, "skip_error", skip_error,
     "stepsizes", stepsizes, "mini_batch_sizes", mini_batch_sizes, 
     "empcomplex", empcomplex);

## Checking that all simulations reached tolerance
fails = [OUTPUTS[i].fail for i=1:length(stepsizes)*numsimu];
if all(s->(string(s)=="tol-reached"), fails)
    println("Tolerance always reached")
end

## Plotting one SAGA-nice simulation for each mini-batch size
if numsimu == 1
    gr()
    # pyplot()
    plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp3.2_test"); # Plot and save output
end

# @printf "\nmethod name      | %s |     %s     |    %s    |  %s   |\n" method_names[1] method_names[2] method_names[3] method_names[4]
# @printf "mini-batch size  |            %d           |       %d      |      %d      |      %d      |\n\n" mini_batch_sizes[1] mini_batch_sizes[2] mini_batch_sizes[3] mini_batch_sizes[4]
# @printf "step size        |       %e      | %e | %e | %e |\n\n" stepsizes[1] stepsizes[2] stepsizes[3] stepsizes[4]
# @printf "total complexity |       %s        |   %s  |  %s  |  %s |\n\n" format(empcomplex[1], commas=true) format(empcomplex[2], commas=true) format(empcomplex[3], commas=true) format(empcomplex[4], commas=true)

@printf "\nmethod name      | %s |     %s     |    %s    |  %s   |  %s   |\n" method_names[1] method_names[2] method_names[3] method_names[4] method_names[5]
@printf "mini-batch size  |            %d           |       %d      |      %d      |      %d      |      %d      |\n" mini_batch_sizes[1] mini_batch_sizes[2] mini_batch_sizes[3] mini_batch_sizes[4] mini_batch_sizes[5]
@printf "step size        |       %e      | %e | %e | %e | %e |\n" stepsizes[1] stepsizes[2] stepsizes[3] stepsizes[4] stepsizes[5]
@printf "total complexity |       %s        |   %s  |  %s  |  %s |  %s |\n\n" format(empcomplex[1], commas=true) format(empcomplex[2], commas=true) format(empcomplex[3], commas=true) format(empcomplex[4], commas=true) format(empcomplex[5], commas=true)


## for the skip_error parameter:
"""
    closest_power_of_ten(integer::Int64)

    Compute the closest power of ten of an integer.

#INPUTS:\\
    - **Int64** integer: integer\\
#OUTPUTS:\\
    - **Int64** or **Float64** closest_power: closest power of ten of the input

# Examples
```jldoctest
julia> closest_power(0)
1
julia> closest_power(9)
1
julia> closest_power(204)
100
```
"""
function closest_power_of_ten(integer::Int64)
    if integer < 0
        closest_power = 10.0 ^ (1 - length(string(integer)));
    else 
        closest_power = 10 ^ (length(string(integer)) - 1);
    end
    return closest_power
end

typeof(closest_power_of_ten(4))
closest_power_of_ten(409)
typeof(closest_power_of_ten(-11))