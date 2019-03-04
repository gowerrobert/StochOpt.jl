"""
### "Optimal mini-batch and step sizes for SAGA", Nidham Gazagnadou, Robert M. Gower, Joseph Salmon (2019)

## --- EXPERIMENT 4 (parallel implementation) ---
Goal: Testing the optimality of our optimal mini-batch size tau_practical with corresponding step size gamma_practical

## --- THINGS TO CHANGE BEFORE RUNNING ---
- line 37: enter your full path to the "StochOpt.jl/" repository in the *path* variable

## --- HOW TO RUN THE CODE ---
To run only the first experiment (ijcnn1_full + column-scaling + lambda=1e-1), open a terminal, go into the "StochOpt.jl/" repository and run the following command:
>julia -p <number_of_processor_to_add> repeat_paper_experiments/repeat_optimal_minibatch_step_sizes_SAGA_paper_experiment_4_parallel.jl false
where <number_of_processor_to_add> has to be replaced by the user.
To launch all the 12 experiments of the paper change the bash input and run:
>julia -p <number_of_processor_to_add> repeat_paper_experiments/repeat_optimal_minibatch_step_sizes_SAGA_paper_experiment_4_parallel.jl true
where <number_of_processor_to_add> has to be replaced by the user.

## --- EXAMPLE OF RUNNING TIME ---
Running time of the first experiment when adding 4 processors on a laptop with 16Gb RAM and Intel® Core™ i7-8650U CPU @ 1.90GHz × 8
88.395703 seconds (2.01 M allocations: 98.863 MiB, 0.05% gc time), around 1min 28s
Running time of all experiments when adding 4 processors on a laptop with 16Gb RAM and Intel® Core™ i7-8650U CPU @ 1.90GHz × 8
XXXX.XXXX seconds (XX.XX G allocations: XX.XX TiB, XX.XX% gc time), around XX.XX

## --- SAVED FILES ---
For each problem (data set + scaling process + regularization),
- the empirical total complexity v.s. mini-batch size plots are saved in ".pdf" format in the "./figures/" folder
- the results of the simulations (mini-batch grid, empirical complexities, optimal empirical mini-batch size, etc.) are saved in ".jld" format in the "./data/" folder
"""


## Bash input
allexperiments = parse(Bool, ARGS[1]); # run 1 (false) or all the 12 experiments (true)

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

## Experiments settings
numsimu = 1; # number of runs of mini-batch SAGA for averaging the empirical complexity
if allexperiments
    experiments = 1:12;
else
    experiments = 1:1;
end

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

@time begin
@sync @distributed for exp in experiments
    data = datasets[exp];
    scaling = scalings[exp];
    lambda = lambdas[exp];
    println("EXPERIMENT : ", exp, " over ", length(experiments));
    @printf "Inputs: %s + %s + %1.1e \n" data scaling lambda;

    Random.seed!(1);
    # Random.seed!(2222);

    ## Loading the data
    println("--- Loading data ---");
    data_path = "$(path)data/";
    X, y = loadDataset(data_path, data);

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
        ## Correcting for logistic since phi'' <= 1/4
        L /= 4;
        # Lmax /= 4;
        # Lbar /= 4;
    end

    ## Computing mini-batch and step sizes
    # tau_simple = round(Int, 1 + (mu*(n-1))/(4*Lbar))
    # tau_bernstein = max(1, round(Int, 1 + (mu*(n-1))/(8*L) - (4/3)*log(d)*((n-1)/n)*(Lmax/(2*L))))
    tau_practical = round(Int, 1 + (mu*(n-1))/(4*L))

    ## Computing the empirical mini-batch size over a grid
    # minibatchgrid = vcat(2 .^ collect(0:7), 2 .^ collect(8:2:floor(Int, log2(n))))
    if data == "real-sim"
        minibatchlist = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14, 2^16];
    else
        minibatchlist = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14];
    end

    println("---------------------------------- MINI-BATCH GRID ------------------------------------------");
    println(minibatchgrid);
    println("---------------------------------------------------------------------------------------------");

    OUTPUTS, itercomplex = simulate_SAGA_nice(prob, minibatchgrid, options, numsimu, skip_multiplier=skip_multiplier[exp]);

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
    if numsimu == 1
        save("$(data_path)$(savename).jld",
        "options", options, "minibatchgrid", minibatchgrid,
        "itercomplex", itercomplex, "empcomplex", empcomplex,
        "tau_empirical", tau_empirical);
    end

    ## Plotting total complexity vs mini-batch size
    pyplot()
    plot_empirical_complexity(prob, minibatchgrid, empcomplex, tau_practical, tau_empirical, path=path)

    println("Practical optimal mini-batch = ", tau_practical)
    println("Empirical optimal mini-batch = ", tau_empirical, "\n\n")
end
end