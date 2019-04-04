"""
### "Optimal mini-batch and step sizes for SAGA", Nidham Gazagnadou, Robert M. Gower, Joseph Salmon (2019)

## --- EXPERIMENTS 1 and 2 (parallel implementation) ---
Goal: Computing the upper-bounds of the expected smoothness constant (exp. 1) and our step sizes estimates (exp. 2).

## --- THINGS TO CHANGE BEFORE RUNNING ---
- line XXXXXXXXXXXX: enter your full path to the "StochOpt.jl/" repository in the *path* variable

## --- HOW TO RUN THE CODE ---
To run only the first 2 problems (ridge regression for gauss-50-24-0.0_seed-1 with lambda=10^(-1) with and without scaling), open a terminal, go into the "StochOpt.jl/" repository and run the following command:
>julia -p <number_of_processor_to_add> repeat_paper_experiments/repeat_optimal_minibatch_step_sizes_SAGA_paper_experiment_1_and_2_parallel.jl false
where <number_of_processor_to_add> has to be replaced by the user.
To launch all the 42 problems of the paper change the bash input and run:
>julia -p <number_of_processor_to_add> repeat_paper_experiments/repeat_optimal_minibatch_step_sizes_SAGA_paper_experiment_1_and_2_parallel.jl true
where <number_of_processor_to_add> has to be replaced by the user.

## --- EXAMPLE OF RUNNING TIME ---
Running time of the first 2 problems (ridge regression for gauss-50-24-0.0_seed-1 with lambda=10^(-1) with and without scaling) when adding 4 processors on a laptop with 16Gb RAM and Intel® Core™ i7-8650U CPU @ 1.90GHz × 8
XXXXXXXXXXXXXXXXX, around XXXXXXX
Running time of all 42 problems when adding 4 processors on a laptop with 16Gb RAM and Intel® Core™ i7-8650U CPU @ 1.90GHz × 8
XXXXXXXXXXXXXXXXX, around XXXXXXX

## --- SAVED FILES ---
For each problem (data set + scaling process + regularization)
- the plots of the upper-bounds of the expected smoothness constant (exp.1) and the ones of the step sizes estimates are saved in ".pdf" format in the "./figures/" folder
- the results of the simulations (smoothness constants, upper-bounds of the expected smoothness constant, estimates of the step sizes and optimal mini-batch estimates) are saved in ".jld" format in the "./data/" folder using function ``save_SAGA_nice_constants``
"""

## Bash input
all_problems = parse(Bool, ARGS[1]); # run 1 (false) or all the 12 problems (true)

using Distributed

@everywhere begin
    path = "/home/nidham/phd/StochOpt.jl/"; # Change the full path here

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

    include("$(path)src/StochOpt.jl")
    # gr()
    pyplot() # No problem with pyplot when called in @everywhere statement
end

data_path = "$(path)data/";

## Experiments settings
numsimu = 1; # number of runs of mini-batch SAGA for averaging the empirical complexity
if all_problems
    datasets = ["gauss-50-24-0.0_seed-1",
                "diagints-24-0.0-100_seed-1",
                "diagalone-24-0.0-100_seed-1",
                "diagints-24-0.0-100-rotated_seed-1",
                "diagalone-24-0.0-100-rotated_seed-1",
                "slice",
                "YearPredictionMSD_full",
                "covtype_binary",
                "rcv1_full",
                "news20_binary",
                "real-sim",
                "ijcnn1_full"]
    lambdas = [10^(-1), 10^(-3)]
    num_problems = 42
else
    datasets = ["gauss-50-24-0.0_seed-1"]
    lambdas = [10^(-1)]
    num_problems = 2
end

@time begin
run_number = 1;
@sync @distributed for idx_prob in problems
    data = datasets[idx_prob];
    scaling = scalings[idx_prob];
    lambda = lambdas[idx_prob];
    skip_error = skip_errors[idx_prob];
    println("\n--- EXPERIMENT : ", idx_prob, " over ", length(problems), " ---\nInputs: ", data, " + ", scaling, " + ", lambda);

    Random.seed!(1);

    ## Loading the data
    println("--- Loading data ---");
    data_path = "$(path)data/";
    X, y = loadDataset(data_path, data);

    ## Setting up the problem
    println("\n--- Setting up the selected problem ---");
    options = set_options(tol=precision, max_iter=10^8, max_epocs=600,
                          max_time=60.0*60.0*5.0,
                          skip_error_calculation=10^5,
                          batchsize=1,
                          regularizor_parameter = "normalized",
                          initial_point="zeros", # is fixed not to add more randomness
                          force_continue=true); # force continue if diverging or if tolerance reached
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
    Lmax = prob.Lmax;
    L = prob.L;
    # Lbar = prob.Lbar;


    #=
        TO FILL IN
    =#





















end
end

println("\n\n--- EXPERIMENTS 1 AND 2 (PARALLEL) FINISHED ---");