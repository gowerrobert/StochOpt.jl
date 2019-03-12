"""
### "Optimal mini-batch and step sizes for SAGA", Nidham Gazagnadou, Robert M. Gower, Joseph Salmon (2019)

## --- EXPERIMENT 4 (parallel implementation) ---
Goal: Testing the optimality of our optimal mini-batch size b_practical with corresponding step size gamma_practical

## --- THINGS TO CHANGE BEFORE RUNNING ---
- line 37: enter your full path to the "StochOpt.jl/" repository in the *path* variable

## --- HOW TO RUN THE CODE ---
To run only the first experiment (ijcnn1_full + column-scaling + lambda=1e-1), open a terminal, go into the "StochOpt.jl/" repository and run the following command:
>julia -p <number_of_processor_to_add> repeat_paper_experiments/repeat_optimal_minibatch_step_sizes_SAGA_paper_experiment_4_parallel.jl false
where <number_of_processor_to_add> has to be replaced by the user.
To launch all the 12 problems of the paper change the bash input and run:
>julia -p <number_of_processor_to_add> repeat_paper_experiments/repeat_optimal_minibatch_step_sizes_SAGA_paper_experiment_4_parallel.jl true
where <number_of_processor_to_add> has to be replaced by the user.

## --- EXAMPLE OF RUNNING TIME ---
Running time of the first experiment when adding 4 processors on a laptop with 16Gb RAM and Intel® Core™ i7-8650U CPU @ 1.90GHz × 8
96.882105 seconds (2.02 M allocations: 99.035 MiB, 0.5% gc time), around 1min 37s
Running time of all problems when adding 4 processors on a laptop with 16Gb RAM and Intel® Core™ i7-8650U CPU @ 1.90GHz × 8
XXXX.XXXX seconds (XX.XX G allocations: XX.XX TiB, XX.XX% gc time), around XX.XX

## --- SAVED FILES ---
For each problem (data set + scaling process + regularization)
- the empirical total complexity v.s. mini-batch size plots are saved in ".pdf" format in the "./figures/" folder
- the results of the simulations (mini-batch grid, empirical complexities, optimal empirical mini-batch size, etc.) are saved in ".jld" format in the "./data/" folder
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

## Experiments settings
numsimu = 1; # number of runs of mini-batch SAGA for averaging the empirical complexity
if all_problems
    problems = 1:12;
else
    problems = 6:6; # DO NOT FORGET TO SET IT BACK TO "1:1"
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
skip_multiplier = [0.01,        # GOOD
                   0.01,        # GOOD
                   0.01,        # GOOD
                   0.01,        # GOOD (1->4 16 min)
                   0.05,        # GOOD 15 min for 0.05
                   0.1,         # 25 min avec 1.0 et 2^18 et n en plus / XXX avec 0.1
                   1.0,         # 22 min avec 1.0 / 43 min avec 0.1
                   1.0,         # 52 min avec 1.0
                   0.1,         # 12 min avec 1.0 / 21 min avec 0.1
                   0.1,         # plus de 7h (max_time reached for b=2^14) avec 1.0 / à lancer avec 0.1
                   0.1,         # 3h 5min avec 1.0 / 3h avec 0.1 / 2h30 avec 0.1
                   0.1];        # 6h30 avec 0.1  / 6h 20 min avec 0.1

precision = 10.0^(-4)

@time begin
@sync @distributed for idx_prob in problems
    data = datasets[idx_prob];
    scaling = scalings[idx_prob];
    lambda = lambdas[idx_prob];
    println("EXPERIMENT : ", idx_prob, " over ", length(problems));
    @printf "Inputs: %s + %s + %1.1e \n" data scaling lambda;

    Random.seed!(1);

    ## Loading the data
    println("--- Loading data ---");
    data_path = "$(path)data/";
    X, y = loadDataset(data_path, data);

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
    d = prob.numfeatures;
    mu = prob.mu;
    # Lmax = prob.Lmax;
    L = prob.L;
    # Lbar = prob.Lbar;

    ## Computing mini-batch and step sizes
    # b_simple = round(Int, 1 + (mu*(n-1))/(4*Lbar))
    # b_bernstein = max(1, round(Int, 1 + (mu*(n-1))/(8*L) - (4/3)*log(d)*((n-1)/n)*(Lmax/(2*L))))
    b_practical = max(1, min(round(Int, 1 + (mu*(n-1))/(4*L)), n)) # Thresholds the optimal mini-batch size when the problem is ill-conditioned

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

    OUTPUTS, itercomplex = simulate_SAGA_nice(prob, minibatchgrid, options, numsimu, skip_multiplier=skip_multiplier[idx_prob]);

    ## Checking that all simulations reached tolerance
    fails = [OUTPUTS[i].fail for i=1:length(minibatchgrid)*numsimu];
    if all(s->(string(s)=="tol-reached"), fails)
        println("Tolerance always reached")
    else
        println("Some total complexities might be threshold because of reached maximal time")
    end

    ## Computing the empirical complexity
    empcomplex = reshape(minibatchgrid .* itercomplex, length(minibatchgrid)); # mini-batch size times number of iterations
    min_empcomplex, idx_min = findmin(empcomplex)
    b_empirical = minibatchgrid[idx_min]

    ## Saving the result of the simulations
    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
    savename = string(probname, "-exp4-optimality-", numsimu, "-avg");
    if numsimu == 1
        save("$(data_path)$(savename).jld",
        "options", options, "minibatchgrid", minibatchgrid,
        "itercomplex", itercomplex, "empcomplex", empcomplex,
        "b_empirical", b_empirical);
    end

    ## Plotting total complexity vs mini-batch size
    pyplot()
    plot_empirical_complexity(prob, minibatchgrid, empcomplex, b_practical, b_empirical, path=path)

    println("Practical optimal mini-batch = ", b_practical)
    println("Empirical optimal mini-batch = ", b_empirical, "\n\n")
end
end

println("\n\n--- EXPERIMENT 4 (PARALLEL) FINISHED ---");