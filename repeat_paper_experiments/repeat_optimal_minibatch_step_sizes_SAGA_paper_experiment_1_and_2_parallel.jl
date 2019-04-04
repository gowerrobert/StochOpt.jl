"""
### "Optimal mini-batch and step sizes for SAGA", Nidham Gazagnadou, Robert M. Gower, Joseph Salmon (2019)

## --- EXPERIMENTS 1 and 2 (parallel implementation) ---
Goal: Computing the upper-bounds of the expected smoothness constant (exp. 1) and our step sizes estimates (exp. 2).

## --- THINGS TO CHANGE BEFORE RUNNING ---
- line 36: enter your full path to the "StochOpt.jl/" repository in the *path* variable

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


## Experiments settings
if all_problems
    problems = 1:42;
else
    problems = 1:1;
end

datasets = collect(Iterators.flatten([fill("gauss-50-24-0.0_seed-1", 4),
                                      fill("diagints-24-0.0-100_seed-1", 4),
                                      fill("diagalone-24-0.0-100_seed-1", 4),
                                      fill("diagints-24-0.0-100-rotated_seed-1", 4),
                                      fill("diagalone-24-0.0-100-rotated_seed-1", 4),
                                      fill("ijcnn1_full", 4),
                                      fill("YearPredictionMSD_full", 4),
                                      fill("covtype_binary", 4),
                                      fill("slice", 4),
                                      fill("real-sim", 2),
                                      fill("rcv1_full", 2),
                                      fill("news20_binary", 2)]))

scalings = collect(Iterators.flatten([Iterators.flatten(fill(["none", "none", "column-scaling", "column-scaling"], 9)),
                                      Iterators.flatten(fill(["none", "none"], 3))]))

lambdas = collect(Iterators.flatten(fill([10^(-1), 10^(-3)], 21)))

data_path = "$(path)data/";

@time begin
@sync @distributed for idx_prob in problems
    data = datasets[idx_prob];
    scaling = scalings[idx_prob];
    lambda = lambdas[idx_prob];
    println("\n--- EXPERIMENT : ", idx_prob, " over ", length(problems), " ---\nInputs: ", data, " + ", scaling, " + ", lambda);

    Random.seed!(1);

    ## Loading the data
    println("--- Loading data ---");
    data_path = "$(path)data/";
    X, y = loadDataset(data_path, data);

    ## Setting up the problem
    println("\n--- Setting up the selected problem ---");
    options = set_options(tol=10.0^(-1), max_iter=10^8, max_time=10.0^2, max_epocs=10^8,
                                  regularizor_parameter = "normalized",
                                  initial_point="zeros", # is fixed not to add more randomness
                                  force_continue=false); # if true, forces continue if diverging or if tolerance reached
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
    datathreshold = 24; # if n is too large we do not compute the exact expected smoothness constant nor corresponding step sizes

    ########################### EMPIRICAL UPPER BOUNDS OF THE EXPECTED SMOOTHNESS CONSTANT ###########################
    ## Computing the upper-bounds
    expsmoothcst = nothing;
    simple_bound, bernstein_bound, practical_approx, expsmoothcst = get_expected_smoothness_bounds(prob);

    ### PLOTTING ###
    println("\n--- Plotting upper-bounds ---");
    pyplot()
    plot_expected_smoothness_bounds(prob, simple_bound, bernstein_bound, practical_approx, expsmoothcst, showlegend=false);

    ## Practical approximation equals true expected smoothness constant for b=1 and b=n as expected, but is not an upper-bound
    if n <= datathreshold
        println("\nPractical - Expected smoothness gap: ", practical_approx - expsmoothcst)
        println("Simple - Practical gap: ", simple_bound - practical_approx)
        println("Bernstein - Simple gap: ", bernstein_bound - simple_bound, "\n")
    end
    ##################################################################################################################


    ##################################### EMPIRICAL UPPER BOUNDS OF THE STEP SIZES ####################################
    ## Computing the step sizes
    simple_step_size, bernstein_step_size, practical_step_size, hofmann_step_size, expsmooth_step_size = get_stepsize_bounds(prob, simple_bound, bernstein_bound, practical_approx, expsmoothcst);

    ## Plotting
    println("\n--- Plotting step sizes ---");
    ## WARNING: there is still a problem of ticking non integer on the xaxis
    pyplot()
    plot_stepsize_bounds(prob, simple_step_size, bernstein_step_size, practical_step_size, hofmann_step_size, expsmooth_step_size, showlegend=false);
    ##################################################################################################################

    ########################################### SAVNG RESULTS ########################################################
    println("\n--- Saving the bounds ---");
    save_SAGA_nice_constants(prob, data, simple_bound, bernstein_bound, practical_approx, expsmoothcst,
                             simple_step_size, bernstein_step_size, practical_step_size, expsmooth_step_size);
    ##################################################################################################################

end
end

println("\n\n--- EXPERIMENTS 1 AND 2 (PARALLEL) FINISHED ---");