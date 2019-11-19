# Replot exp1.b SVRG for slice + 10^-3

## Detail in file names
details = "test-rho"
problems = [6]

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
using SharedArrays

## Bash input
machine = "home" # ARGS[1]
if machine == "lame10"
    path = "/cal/homes/ngazagnadou/StochOpt.jl/"   # lame10
elseif machine == "lame23"
    path = "/home/infres/ngazagnadou/StochOpt.jl/" # lame23
elseif machine == "home"
    path = "/home/nidham/phd/StochOpt.jl/"         # local
end
println("path: ", path)

include("$(path)src/StochOpt.jl")
pyplot()

## Path settings
folder_path = "$(path)experiments/theory_practice_SVRG/exp1b/"

## Experiments settings

datasets = ["ijcnn1_full", "ijcnn1_full",                       # scaled,         n = 141,691, d =     22
            "YearPredictionMSD_full", "YearPredictionMSD_full", # scaled,         n = 515,345, d =     90
            "slice", "slice",                                   # scaled,         n =  53,500, d =    384
            "real-sim", "real-sim"]                             # unscaled,       n =  72,309, d = 20,958


scalings = ["column-scaling", "column-scaling",
            "column-scaling", "column-scaling",
            "column-scaling", "column-scaling",
            "none", "none"]

lambdas = [10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3)]

## Set smaller number of skipped iteration for finer estimations (yet, longer simulations)
skip_errors = [[700 7000 -2. 7000],     # 1)  ijcnn1_full + scaled + 1e-1                 midnight retry / FINAL
               [13000 7000 -2. 5000],   # 2)  ijcnn1_full + scaled + 1e-3               midnight retry / FINAL
               [50000 30000 -2. 20000], # 3)  YearPredictionMSD_full + scaled + 1e-1  midnight retry / FINAL
               [60000 15000 -2. 5000],  # 4)  YearPredictionMSD_full + scaled + 1e-3   midnight retry / FINAL
               [50000 2500 -2. 2500],   # 5)  slice + scaled + 1e-1                     100 epochs / FINAL
               [50000 2500 -2. 1],      # 6)  slice + scaled + 1e-3                        100 epochs / FINAL
               [  10 2000 -2. 4000],    # 7)  real-sim + unscaled + 1e-1                 midnight retry / FINAL
               [500 5000 -2. 2000]]     # 8)  real-sim + unscaled + 1e-3                   midnight retry / FINAL

idx_prob = 6
# for idx_prob in problems
    data = datasets[idx_prob]
    scaling = scalings[idx_prob]
    lambda = lambdas[idx_prob]

    ## Loading the data
    println("--- Loading data ---")
    data_path = "$(path)data/";
    X, y = loadDataset(data_path, data)

    ## Setting up the problem
    println("\n--- Setting up the selected problem ---")
    options = set_options(tol=10.0^(-6), max_iter=10^8, max_epocs=10^4, max_time=60.0, skip_error_calculation=10^5, batchsize=1,
                          regularizor_parameter="normalized", initial_point="zeros", force_continue=false)

    u = unique(y)
    if length(u) < 2
        error("Wrong number of possible outputs")
    elseif length(u) == 2
        println("Binary output detected: the problem is set to logistic regression")
        prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling=scaling)
    else
        println("More than three modalities in the outputs: the problem is set to ridge regression")
        prob = load_ridge_regression(X, y, data, options, lambda=lambda, scaling=scaling)
    end

    ## Loading data
    suffix = "lame23"
    filename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
    filename = string(filename, "-exp1b-$(suffix)-$(details)")
    OUTPUTS = load("$(folder_path)data/$(filename).jld", "OUTPUTS")


    ## Replot: only change legend position
    pyplot()
    save_path = "$(path)experiments/theory_practice_SVRG/exp1b/"
    plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp1b-$(suffix)-$(details)", path=save_path, legendpos=:best, legendfont=8)
# end

