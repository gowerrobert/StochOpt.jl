"""
### "Optimal mini-batch and step sizes for SAGA", Nidham Gazagnadou, Robert M. Gower, Joseph Salmon (2019)

## --- EXPERIMENTS 1 and 2 (serial implementation) ---
Goal: Computing the upper-bounds of the expected smoothness constant (exp. 1) and our step sizes estimates (exp. 2).

## --- THINGS TO CHANGE BEFORE RUNNING ---

## --- HOW TO RUN THE CODE ---
To run only the first problem (XXXX), open a terminal, go into the "StochOpt.jl/" repository and run the following command:
>julia repeat_paper_experiments/repeat_optimal_minibatch_step_sizes_SAGA_paper_experiment_1_and_2.jl false
To launch all the 42 problems of the paper change the bash input and run:
>julia repeat_paper_experiments/repeat_optimal_minibatch_step_sizes_SAGA_paper_experiment_1_and_2.jl true

## --- EXAMPLE OF RUNNING TIME ---
Running time of the first problem on a laptop with 16Gb RAM and Intel® Core™ i7-8650U CPU @ 1.90GHz × 8
, around XXmin XXs
Running time of all 42 problems on a laptop with 16Gb RAM and Intel® Core™ i7-8650U CPU @ 1.90GHz × 8
, around 1h 20min ## TO DOUBLE CHECK

## --- SAVED FILES ---
For each problem (data set + scaling process + regularization)
- the plots of the upper-bounds of the expected smoothness constant (exp.1) and the ones of the step sizes estimates are saved in ".pdf" format in the "./figures/" folder
- the results of the simulations (smoothness constants, upper-bounds of the expected smoothness constant, estimates of the step sizes and optimal mini-batch estimates) are saved in ".jld" format in the "./data/" folder using function ``save_SAGA_nice_constants``
"""

# ## Bash input
# all_problems = parse(Bool, ARGS[1]); # run 1 (false) or all the 42 problems (true)

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
using LaTeXStrings

## Manual inputs
include("../src/StochOpt.jl") # Be carefull about the path here
default_path = "./data/";

# if all_problems
#     problems = 1:42;
# else
#     problems = 1:1;
# end

# datasets = ["ijcnn1_full", "ijcnn1_full", # scaled
#             "YearPredictionMSD_full", "YearPredictionMSD_full", # scaled
#             "covtype_binary", "covtype_binary", # scaled
#             "slice", "slice", # scaled
#             "slice", "slice", # unscaled
#             "real-sim", "real-sim"]; # unscaled

# scalings = ["column-scaling", "column-scaling",
#             "column-scaling", "column-scaling",
#             "column-scaling", "column-scaling",
#             "column-scaling", "column-scaling",
#             "none", "none",
#             "none", "none"];

# lambdas = [10^(-1), 10^(-3),
#            10^(-1), 10^(-3),
#            10^(-1), 10^(-3),
#            10^(-1), 10^(-3),
#            10^(-1), 10^(-3),
#            10^(-1), 10^(-3)];

datasets = ["gauss-50-24-0.0_seed-1"
            "diagints-24-0.0-100_seed-1"
            "diagalone-24-0.0-100_seed-1"
            "diagints-24-0.0-100-rotated_seed-1"
            "diagalone-24-0.0-100-rotated_seed-1"
            "slice"
            "YearPredictionMSD_full"
            "covtype_binary"
            "rcv1_full"
            "news20_binary"
            "real-sim"
            "ijcnn1_full"]

scalings = ["none" "column-scaling"] #for all datasets except real-sim, news20.binary and rcv1

lambdas = [10^(-3) 10^(-1)];

## Small example of relaunched experiment
# datasets = ["ijcnn1_full"]
# scalings = ["none"]
# lambdas = [10^(-1)];

run_number = 1;
for data in datasets
    for lambda in lambdas
        if !(data in ["real-sim" "news20_binary" "rcv1_full"])
            scalings = ["none" "column-scaling"];
        else
            scalings = ["none"];
        end

        for scaling in scalings
            println("\n\n######################################################################")
            println("Run ", string(run_number), " over 42");
            println("Dataset: ", data);
            @printf "lambda: %1.0e\n" lambda;
            println("scaling: ", scaling);
            println("######################################################################\n")

            ## Loading data
            println("--- Loading data ---");
            X, y = loadDataset(default_path, data); # Only loading existing datasets, no data generation here

            ## Setting up the problem
            println("\n--- Setting up the selected problem ---");
            options = set_options(tol=10.0^(-1), max_iter=10^8, max_time=10.0^2, max_epocs=10^8,
                                  regularizor_parameter = "normalized",
                                  initial_point="zeros", # is fixed not to add more randomness
                                  force_continue=false); # if true, forces continue if diverging or if tolerance reached
            u = unique(y);
            if length(u) < 2
                error("Wrong number of possible outputs")
            elseif length(u) == 2
                println("Binary output detected: the problem is set to logistic regression")
                prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling=scaling);
            else
                println("More than three modalities in the outputs: the problem is set to ridge regression")
                prob = load_ridge_regression(X, y, data, options, lambda=lambda, scaling=scaling);
            end

            n = prob.numdata;
            datathreshold = 24; # if n is too large we do not compute the exact expected smoothness constant nor corresponding step sizes

            ########################### EMPIRICAL UPPER BOUNDS OF THE EXPECTED SMOOTHNESS CONSTANT ###########################
            ## Computing the upper-bounds
            expsmoothcst = nothing;
            simple_bound, bernstein_bound, practical_approx, expsmoothcst = get_expected_smoothness_bounds(prob);

            ### PLOTING ###
            println("\n--- Ploting upper-bounds ---");
            pyplot()
            plot_expected_smoothness_bounds(prob, simple_bound, bernstein_bound, practical_approx, expsmoothcst, showlegend=false);

            ## Practical approximation equals true expected smoothness constant for b=1 and b=n as expected, but is not an upper-bound
            if n <= datathreshold
                println("\nPractical - expected smoothness gap: ", practical_approx - expsmoothcst)
                println("Simple - Practical gap: ", simple_bound - practical_approx)
                println("Bernstein - Simple gap: ", bernstein_bound - simple_bound, "\n")
            end
            ##################################################################################################################


            ##################################### EMPIRICAL UPPER BOUNDS OF THE STEP SIZES ####################################
            ## Computing the step sizes
            simple_step_size, bernstein_step_size, practical_step_size, hofmann_step_size, expsmooth_step_size = get_stepsize_bounds(prob, simple_bound, bernstein_bound, practical_approx, expsmoothcst);

            ## Plotting
            println("\n--- Ploting step sizes ---");
            ## WARNING: there is still a problem of ticking non integer on the xaxis
            pyplot()
            plot_stepsize_bounds(prob, simple_step_size, bernstein_step_size, practical_step_size, hofmann_step_size, expsmooth_step_size, showlegend=false);
            ##################################################################################################################

            ########################################### SAVNG RESULTS ########################################################
            save_SAGA_nice_constants(prob, data, simple_bound, bernstein_bound, practical_approx, expsmoothcst,
                                     simple_step_size, bernstein_step_size, practical_step_size, expsmooth_step_size);
            ##################################################################################################################
            global run_number += 1;
        end
    end
end

println("\n\n--- EXPERIMENTS 1 AND 2 FINISHED ---");
