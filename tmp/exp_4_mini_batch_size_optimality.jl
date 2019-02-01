### EXPERIMENT 4

## Testing the optimality of our tau* for the same gamma = gamma_heuristic

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
using LaTeXStrings

## Bash inputs
include("../src/StochOpt.jl") # Be carefull about the path here
data = ARGS[1];
scaling = ARGS[2];
lambda = parse(Float64, ARGS[3]);
println("Inputs: ", data, " + ", scaling, " + ",  lambda, "\n");

## Manual inputs
# include("./src/StochOpt.jl") # Be carefull about the path here
# datasets = readlines("$(default_path)available_datasets.txt");
# idx = 6;
# data = datasets[idx];
# scaling = "none";
# # scaling = "column-scaling";
# lambda = 10^(-1);
# # lambda = 10^(-3);

default_path = "./data/";

Random.seed!(2222);

### LOADING THE DATA ###
println("--- Loading data ---");
X, y = loadDataset(default_path, data);

######################################## SETTING UP THE PROBLEM ########################################
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-4), max_iter=10^8, max_epocs=10^8,
                      max_time=60.0*60.0*3.0,
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

if occursin("lgstc", prob.name) # julia 0.7
    ## Correcting for logistic since phi'' <= 1/4
    Lmax /= 4;
end

### Plot of the empirical complexity vs tau (averaged runs) ###
# tau_simple = round(Int, 1 + (mu*(n-1))/(4*Lbar))
# tau_bernstein = max(1, round(Int, 1 + (mu*(n-1))/(8*L) - (4/3)*log(d)*((n-1)/n)*(Lmax/(2*L))))
tau_heuristic = round(Int, 1 + (mu*(n-1))/(4*L))

######################################## EMPIRICAL OPTIMAL MINIBATCH SIZE ########################################
## Empirical stepsizes returned by optimal mini-batch SAGA with line searchs

# minibatchlist = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14];

minibatchlist = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14, 2^16];

println("---------------------------------- MINI-BATCH LIST ------------------------------------------");
println(minibatchlist);
println("---------------------------------------------------------------------------------------------");

numsimu = 1; # number of runs of mini-batch SAGA for averaging the empirical complexity
@time OUTPUTS, itercomplex = simulate_SAGA_nice(prob, minibatchlist, options, numsimu, skip_multiplier=0.000005);

## Checking that all simulations reached tolerance
fails = [OUTPUTS[i].fail for i=1:length(minibatchlist)*numsimu];
if all(s->(string(s)=="tol-reached"), fails)
    println("Tolerance always reached")
end

## Plotting one SAGA-nice simulation for each mini-batch size
# if numsimu == 1
#     # gr()
#     pyplot()
#     plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp4.1"); # Plot and save output
# end

## Computing the empirical complexity
# itercomplex -= 1; #-> should we remove 1 from itercomplex?
empcomplex = reshape(minibatchlist.*itercomplex, length(minibatchlist)); # tau times number of iterations
min_empcomplex, idx_min = findmin(empcomplex)
tau_empirical = minibatchlist[idx_min]

## Saving the result of the simulations
probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
savename = string(probname, "-exp4-optimality-", numsimu, "-avg");
if numsimu == 1
    save("$(default_path)$(savename).jld",
    "options", options, "minibatchlist", minibatchlist,
    "itercomplex", itercomplex, "empcomplex", empcomplex,
    "tau_empirical", tau_empirical);
end

# Remove other estimation of the best tau for bernstein and simple

pyplot()
plot_empirical_complexity(prob, minibatchlist, empcomplex, tau_heuristic, tau_empirical);

println("practical optimal tau = ", tau_heuristic);
println("The empirical optimal tau = ", tau_empirical);

##########################################################################################""


# """
#     compute_skip_error(n::Int64, minibatch_size::Int64, skip_multiplier::Float64=0.02)

# Compute the number of skipped iterations between two error estimation.
# The computation rule is arbitrary, but depends on the dimension of the problem and on the mini-batch size.

# #INPUTS:\\
#     - **Int64** n: number of data samples\\
#     - **Int64** minibatch\\_size: size of the mini-batch\\
#     - **Float64** skip\\_multiplier: arbitrary multiplier\\
# #OUTPUTS:\\
#     - **Int64** skipped_errors: number iterations between two evaluations of the error\\
# """
# function compute_skip_error(n::Int64, minibatch_size::Int64, skip_multiplier::Float64=0.015)
#     tmp = floor(skip_multiplier*n/minibatch_size);
#     skipped_errors = 1;
#     while(tmp > 1.0)
#         tmp /= 2;
#         skipped_errors *= 2^1;
#     end
#     skipped_errors = convert(Int64, skipped_errors); # Seems useless

#     return skipped_errors
# end

# compute_skip_error(n, 2)
# compute_skip_error(n, 64)
# compute_skip_error(n, 4096)


# """
#     closest_power_of_ten(integer::Int64)

#     Compute the closest power of ten of an integer.

# #INPUTS:\\
#     - **Int64** integer: integer\\
# #OUTPUTS:\\
#     - **Int64** or **Float64** closest_power: closest power of ten of the input

# # Examples
# ```jldoctest
# julia> closest_power(0)
# 1
# julia> closest_power(9)
# 1
# julia> closest_power(204)
# 100
# ```
# """
# function closest_power_of_ten(integer::Int64)
#     if integer < 0
#         closest_power = 10.0 ^ (1 - length(string(integer)));
#     else
#         closest_power = 10 ^ (length(string(integer)) - 1);
#     end
#     return closest_power
# end


# closest_power_of_ten.(round.(Int, n ./ (5*2) ))
# closest_power_of_ten.(round.(Int, n ./ (5*64) ))
# closest_power_of_ten.(round.(Int, n ./ (5*4096) ))
