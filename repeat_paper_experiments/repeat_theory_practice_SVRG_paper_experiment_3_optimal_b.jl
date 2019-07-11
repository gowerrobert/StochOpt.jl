"""
### "Towards closing the gap between the theory and practice of SVRG", O. Sebbouh, S. Jelassi, N. Gazagnadou, F. Bach, R. M. Gower (2019)

## --- EXPERIMENT 3 ---
Goal: Testing the optimality of our optimal mini-batch size versus grid-search ones.

## --- THINGS TO CHANGE BEFORE RUNNING ---


## --- HOW TO RUN THE CODE ---
To run this experiment, open a terminal, go into the "StochOpt.jl/" repository and run the following command:
>julia repeat_paper_experiments/repeat_theory_practice_SVRG_paper_experiment_3_optimal_b.jl

## --- EXAMPLE OF RUNNING TIME ---

## --- SAVED FILES ---

"""


## General settings
max_epochs = 10^8
max_time = 60.0*10.0 # 60.0*60.0*24.0
precision = 10.0^(-6)

## File names
# details = "final"
details = "test"

## Bash input
all_problems = parse(Bool, ARGS[1]) # run 1 (false) or all the 8 problems (true)

using Distributed

@everywhere begin
    # path = "/home/nidham/phd/StochOpt.jl/" # Change the full path here
    path = "/home/infres/ngazagnadou/StochOpt.jl/" # lame23

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

if path == "/home/infres/ngazagnadou/StochOpt.jl/"
    suffix = "lame23"
else
    suffix = "home"
end


## Create saving directories if not existing
save_path = "$(path)experiments/theory_practice_SVRG/"
#region
if !isdir(save_path)
    mkdir(save_path)
end
save_path = "$(save_path)exp3/"
if !isdir(save_path)
    mkdir(save_path)
end
if !isdir("$(save_path)data/")
    mkdir("$(save_path)data/")
end
if !isdir("$(save_path)figures/")
    mkdir("$(save_path)figures/")
end
#endregion

## Experiments settings
numsimu = 1; # number of runs of mini-batch SAGA for averaging the empirical complexity
if all_problems
    problems = 1:8
else
    problems = 1:1
end

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

## In the following table, set smaller values for finer estimations (yet, longer simulations)
skip_multipliers = [0.01,  #
                    0.01,  #
                    0.1,  #
                    0.1,  #
                    0.1,  #
                    0.1,  #
                    0.01,  #
                    0.01]  #

@time begin
@sync @distributed for idx_prob in problems
    data = datasets[idx_prob]
    scaling = scalings[idx_prob]
    lambda = lambdas[idx_prob]
    println("EXPERIMENT : ", idx_prob, " over ", length(problems))
    @printf "Inputs: %s + %s + %1.1e \n" data scaling lambda

    Random.seed!(1)

    if idx_prob == 5 || idx_prob == 6
        global precision = 10.0^(-3)
        global max_epochs = 100
    end

    ## Loading the data
    println("--- Loading data ---")
    data_path = "$(path)data/"
    X, y = loadDataset(data_path, data)

    ## Setting up the problem
    println("\n--- Setting up the selected problem ---")
    options = set_options(tol=precision, max_iter=10^8,
                          max_epocs=max_epochs,
                          max_time=max_time,
                          skip_error_calculation=10^4,
                          batchsize=1,
                          regularizor_parameter = "normalized",
                          initial_point="zeros", # is fixed not to add more randomness
                          force_continue=false) # force continue if diverging or if tolerance reached
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

    X = nothing
    y = nothing

    n = prob.numdata
    d = prob.numfeatures
    mu = prob.mu
    Lmax = prob.Lmax
    L = prob.L

    #############################
    numinneriters = n
    b_optimal = optimal_minibatch_Free_SVRG_nice_tight(numinneriters, n, mu, L, Lmax)


    ## Computing the empirical mini-batch size over a grid
    if data == "ijcnn1_full"
        # minibatchgrid = [2^0, 2^5, n] # debugging
        minibatchgrid = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14, 2^16, n]
    elseif data == "YearPredictionMSD_full"
        minibatchgrid = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14, 2^16, 2^18, n]
    elseif data == "slice"
        minibatchgrid = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14, n]
    elseif data == "real-sim"
        minibatchgrid = [2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^14, 2^16, n]
    else
        error("No mini-batch grid available for this dataset")
    end

    println("---------------------------------- MINI-BATCH GRID ------------------------------------------")
    println(minibatchgrid)
    println("---------------------------------------------------------------------------------------------")

    OUTPUTS, itercomplex = simulate_Free_SVRG_nice(prob, minibatchgrid, options, save_path, numsimu=numsimu, skip_multiplier=skip_multipliers[idx_prob], suffix="$(suffix)-$(details)")

    ## Checking that all simulations reached tolerance
    fails = [OUTPUTS[i].fail for i=1:length(minibatchgrid)*numsimu];
    if all(s->(string(s)=="tol-reached"), fails)
        println("Tolerance always reached")
    else
        println("Some total complexities might be threshold because of reached maximal time")
    end

    ## Computing the empirical complexity
    empcomplex = reshape(minibatchgrid .* itercomplex, length(minibatchgrid)) # mini-batch size times number of iterations
    min_empcomplex, idx_min = findmin(empcomplex)
    b_empirical = minibatchgrid[idx_min]

    ## Saving the result of the simulations
    savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
    savename = string(savename, "-exp3-$(numsimu)-avg")
    savename = string(savename, "_skip_mult_", replace(string(skip_multipliers[idx_prob]), "." => "_")) # Extra suffix to check which skip values to keep
    savename = string(savename, "-$(suffix)-$(details)")
    if numsimu == 1
        save("$(save_path)data/$(savename).jld",
        "options", options, "minibatchgrid", minibatchgrid,
        "itercomplex", itercomplex, "empcomplex", empcomplex,
        "b_empirical", b_empirical)
    end

    ## Plotting total complexity vs mini-batch size
    pyplot()
    plot_empirical_complexity(prob, minibatchgrid, empcomplex, b_optimal, b_empirical, path=save_path, skip_multiplier=skip_multipliers[idx_prob], legendpos=:topleft, suffix=suffix)

    println("Practical optimal mini-batch = ", b_optimal)
    println("Empirical optimal mini-batch = ", b_empirical, "\n\n")

end
end

println("\n\n--- EXPERIMENT 3 FINISHED ---")
