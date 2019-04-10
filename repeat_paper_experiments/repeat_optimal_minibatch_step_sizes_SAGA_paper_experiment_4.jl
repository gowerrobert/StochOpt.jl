"""
### "Optimal mini-batch and step sizes for SAGA", Nidham Gazagnadou, Robert M. Gower, Joseph Salmon (2019)

## --- EXPERIMENT 4 (serial implementation) ---
Goal: Testing the optimality of our optimal mini-batch size tau_practical with corresponding step size gamma_practical

## --- THINGS TO CHANGE BEFORE RUNNING ---

## --- HOW TO RUN THE CODE ---
To run only the first problem (ijcnn1_full + column-scaling + lambda=1e-1), open a terminal, go into the "StochOpt.jl/" repository and run the following command:
>julia repeat_paper_experiments/repeat_optimal_minibatch_step_sizes_SAGA_paper_experiment_4.jl false
To launch all the 12 problems of the paper change the bash input and run:
>julia repeat_paper_experiments/repeat_optimal_minibatch_step_sizes_SAGA_paper_experiment_4.jl true

## --- EXAMPLE OF RUNNING TIME ---
Running time of the first problem on a laptop with 16Gb RAM and Intel® Core™ i7-8650U CPU @ 1.90GHz × 8
100.569983 seconds (214.39 M allocations: 52.932 GiB, 9.88% gc time), around 1min 41s
Running time of all problems on a laptop with 16Gb RAM and Intel® Core™ i7-8650U CPU @ 1.90GHz × 8
Too long... many hours (run rather the parallel implementation)
16Gb RAM is not enough for the real-sim dataset (memory might be stored on the swap of the machine which explodes the coputation time)
We encourage users to run the code on servers.

## --- SAVED FILES ---
For each problem (data set + scaling process + regularization)
- the empirical total complexity vs mini-batch size plots are saved in ".pdf" format in the "./figures/" folder
- the results of the simulations (empirical complexities, optimal empirical mini-batch size, etc) are saved in ".jld" format in the "./data/" folder
"""

## Bash input
all_problems = parse(Bool, ARGS[1]); # run 1 (false) or all the 12 problems (true)

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

include("../src/StochOpt.jl")

## Experiments settings
default_path = "./data/";

if all_problems
    problems = 1:12;
else
    problems = 1:1;
end

datasets = ["ijcnn1_full", "ijcnn1_full", # scaled
            "YearPredictionMSD_full", "YearPredictionMSD_full", # scaled
            "covtype_binary", "covtype_binary", # scaled
            "slice", "slice", # scaled
            "slice", "slice", # unscaled
            "real-sim", "real-sim"]; # unscaled

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
skip_multipliers = [0.01,        # 2min 20s avec 0.01
                    0.01,        # 4 min avec 0.01
                    0.01,        # 11 min avec 0.01
                    0.01,        # 10 min avec 0.01
                    0.05,        # 36 min avec 0.05 / 1h 30min avec 0.01 / 5h30 avec 0.001
                    1.0,         # 1h 47min avec 1.0 / 2h 36min avec 0.1
                    0.1,         # 22 min avec 1.0 / 43 min avec 0.1
                    1.0,         # 52 min avec 1.0
                    0.1,         # 12 min avec 1.0 / 21 min avec 0.1
                    1.0,         # plus de 7h (max_time reached for b=2^14) avec 1.0 / 6h30 avec 10.0 (pas assez précis)
                    0.1,         # 1h 27min avec 1.0 (pas assez précis) / 2h 12 min avec 0.1
                    0.1];        # 6h 15min avec 1.0 / more than 6h avec 0.1

precision = 10.0^(-4)

@time begin
for idx_prob in problems
    data = datasets[idx_prob];
    scaling = scalings[idx_prob];
    lambda = lambdas[idx_prob];
    println("EXPERIMENT : ", idx_prob, " over ", length(problems));
    @printf "Inputs: %s + %s + %1.1e \n" data scaling lambda;

    Random.seed!(1);
    # Random.seed!(2222);

    ## Loading the data
    println("--- Loading data ---");
    X, y = loadDataset(default_path, data);

    ## Setting up the problem
    println("\n--- Setting up the selected problem ---");
    options = set_options(tol=precision, max_iter=10^8, max_epocs=10^8,
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
    # d = prob.numfeatures;
    mu = prob.mu;
    # Lmax = prob.Lmax;
    L = prob.L;
    # Lbar = prob.Lbar;

    ## Computing mini-batch and step sizes
    # tau_simple = round(Int, 1 + (mu*(n-1))/(4*Lbar))
    # tau_bernstein = max(1, round(Int, 1 + (mu*(n-1))/(8*L) - (4/3)*log(d)*((n-1)/n)*(Lmax/(2*L))))
    tau_practical = round(Int, 1 + (mu*(n-1))/(4*L))

    ## Computing the empirical mini-batch size over a grid
    # minibatchgrid = vcat(2 .^ collect(0:7), 2 .^ collect(8:2:floor(Int, log2(n))))
    if data == "covtype_binary"
        minibatchgrid = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14, 2^16, 2^18, n]
    elseif data == "ijcnn1_full" && lambda == 10^(-1)
        minibatchgrid = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14, 2^16, n]
    elseif data == "real-sim"
        minibatchgrid = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14, 2^16]
    else
        minibatchgrid = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14]
    end

    println("---------------------------------- MINI-BATCH GRID ------------------------------------------");
    println(minibatchgrid);
    println("---------------------------------------------------------------------------------------------");

    numsimu = 1; # number of runs of mini-batch SAGA for averaging the empirical complexity
    OUTPUTS, itercomplex = simulate_SAGA_nice(prob, minibatchgrid, options, numsimu, skip_multiplier=skip_multiplier[idx_prob]);

    ## Checking that all simulations reached tolerance
    fails = [OUTPUTS[i].fail for i=1:length(minibatchgrid)*numsimu];
    if all(s->(string(s)=="tol-reached"), fails)
        println("Tolerance always reached")
    else
        error("Tolerance is not always reached")
    end

    ## Computing the empirical complexity
    empcomplex = reshape(minibatchgrid .* itercomplex, length(minibatchgrid)); # mini-batch size times number of iterations
    min_empcomplex, idx_min = findmin(empcomplex)
    tau_empirical = minibatchgrid[idx_min]

    ## Saving the result of the simulations
    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
    savename = string(probname, "-exp4-optimality-", numsimu, "-avg");
    # savename = string(savename, "_skip_mult_", replace(string(skip_multipliers[idx_prob]), "." => "_")); # Extra suffix to check which skip values to keep
    if numsimu == 1
        save("$(default_path)$(savename).jld",
        "options", options, "minibatchgrid", minibatchgrid,
        "itercomplex", itercomplex, "empcomplex", empcomplex,
        "tau_empirical", tau_empirical);
    end

    ## Plotting total complexity vs mini-batch size
    pyplot()
    plot_empirical_complexity(prob, minibatchgrid, empcomplex, tau_practical, tau_empirical)

    println("Practical optimal mini-batch = ", tau_practical)
    println("Empirical optimal mini-batch = ", tau_empirical, "\n\n")
end
end

println("\n\n--- EXPERIMENT 4 FINISHED ---");