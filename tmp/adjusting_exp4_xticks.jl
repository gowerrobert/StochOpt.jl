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
using Formatting
using LaTeXStrings


include("./src/StochOpt.jl")

## Experiments settings
default_path = "./data/";

data = "covtype_binary"
scaling = "column-scaling"
lambda = 10^(-1)
@printf "Inputs: %s + %s + %1.1e \n" data scaling lambda;

Random.seed!(1);

## Loading the data
println("--- Loading data ---");
X, y = loadDataset(default_path, data);

## Setting up the problem
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-4), max_iter=10^8, max_epocs=10^8,
                      max_time=60.0*60.0*5.0,
                      skip_error_calculation=10^4,
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
    prob = load_ridge_regression(X, y, data, options, lambda=lambda, scaling=scaling);
end

X = nothing;
y = nothing;

n = prob.numdata;
d = prob.numfeatures;
mu = prob.mu
# Lmax = prob.Lmax;
L = prob.L;

if occursin("lgstc", prob.name)
    ## Correcting for logistic since phi'' <= 1/4 #TOCHANGE
    L /= 4;
end

## Computing mini-batch and step sizes
tau_practical = round(Int, 1 + (mu*(n-1))/(4*L))

probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
savename = string(probname, "-exp4-optimality-", 1, "-avg");
dic = load("$(default_path)$(savename).jld")

options = dic["options"]
itercomplex = dic["itercomplex"]
tau_empirical = dic["tau_empirical"]
minibatchgrid = dic["minibatchgrid"]
empcomplex = dic["empcomplex"]

## Plotting total complexity vs mini-batch size
pyplot()
plot_empirical_complexity(prob, minibatchgrid, empcomplex, tau_practical, tau_empirical)


#################################################################################################

## Double checking if mini-batch grid is correct
using Printf

## Experiments settings
numsimu = 1; # number of runs of mini-batch SAGA for averaging the empirical complexity

problems = 1:12;

datasets = ["ijcnn1_full", "ijcnn1_full",                       # scaled,   n = 141,691, d =     22
            "YearPredictionMSD_full", "YearPredictionMSD_full", # scaled,   n = 515,345, d =     90
            "covtype_binary", "covtype_binary",                 # scaled,   n = 581,012, d =     54
            "slice", "slice",                                   # scaled,   n =  53,500, d =    384
            "slice", "slice",                                   # unscaled, n =  53,500, d =    384
            "real-sim", "real-sim"];                            # unscaled, n =  72,309, d = 20,958

scalings = ["column-scaling", "column-scaling",
            "column-scaling", "column-scaling",
            "column-scaling", "column-scaling",
            "column-scaling", "column-scaling",
            "none", "none",
            "none", "none"];

lambdas = [10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3)];

## In the following table, set smaller values for finer estimations (yet, longer simulations)
skip_multiplier = [0.05, 0.05,
                   0.05, 0.05,
                   0.05, 0.05,
                   0.05, 0.05,
                   0.05, 0.05,
                   0.05, 0.05];

for idx_prob in problems
    data = datasets[idx_prob];
    scaling = scalings[idx_prob];
    lambda = lambdas[idx_prob];
    println("EXPERIMENT : ", idx_prob, " over ", length(problems));
    @printf "Inputs: %s + %s + %1.1e \n" data scaling lambda;

    ## Computing the empirical mini-batch size over a grid
    # minibatchgrid = vcat(2 .^ collect(0:7), 2 .^ collect(8:2:floor(Int, log2(n))))
    if data == "covtype_binary" && lambda == 10^(-1)
        println("up to 2 ^ 18\n")
        minibatchgrid = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14, 2^16, 2^18]
    elseif data == "real-sim" && lambda == 10^(-1)
        println("up to n\n")
        minibatchgrid = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14, 2^16, 72309]
    elseif data == "real-sim" && lambda == 10^(-3)
        println("up to 2 ^ 16\n")
        minibatchgrid = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14, 2^16]
    else
        println("up to 2 ^ 14\n")
        minibatchgrid = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14]
    end
end