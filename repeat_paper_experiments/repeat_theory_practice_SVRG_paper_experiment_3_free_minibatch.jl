"""
### "Towards closing the gap between the theory and practice of SVRG", O. Sebbouh, S. Jelassi, N. Gazagnadou, F. Bach, R. M. Gower (2019)

## --- EXPERIMENT 3 ---
Goal: Comparing Free-SVRG with m=n for different mini-batch sizes {1, 100, \sqrt{n}, n, b^*}.

## --- THINGS TO CHANGE BEFORE RUNNING ---


## --- HOW TO RUN THE CODE ---
To run this experiment, open a terminal, go into the "StochOpt.jl/" repository and run the following command:
>julia repeat_paper_experiments/repeat_theory_practice_SVRG_paper_experiment_3-free_minibatch.jl

## --- EXAMPLE OF RUNNING TIME ---

## --- SAVED FILES ---

"""

## General settings
max_epochs = 10^8
max_time = 60.0*60.0*10.0
precision = 10.0^(-6)

## Bash input
# all_problems = parse(Bool, ARGS[1]) # run 1 (false) or all the 12 problems (true)
# problems = parse.(Int64, ARGS)
machine = ARGS[1]
problems = [parse(Int64, ARGS[2])]
println("problems: ", problems)

using Distributed

@everywhere begin
    if machine == "lame10"
        path = "/cal/homes/ngazagnadou/StochOpt.jl/"   # lame10
    elseif machine == "lame23"
        path = "/home/infres/ngazagnadou/StochOpt.jl/" # lame23
    elseif machine == "home"
        path = "/home/nidham/phd/StochOpt.jl/"         # local
    end
    println("path: ", path)

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
# if all_problems
#     problems = 1:16
# else
#     problems = 1:1
# end

datasets = ["ijcnn1_full", "ijcnn1_full",                       # scaled,         n = 141,691, d =     22
            "YearPredictionMSD_full", "YearPredictionMSD_full", # scaled,         n = 515,345, d =     90
            "covtype_binary", "covtype_binary",                 # scaled,         n = 581,012, d =     54
            "slice", "slice",                                   # scaled,         n =  53,500, d =    384
            "real-sim", "real-sim",                             # unscaled,       n =  72,309, d = 20,958
            "a1a_full", "a1a_full",                             # unscaled,       n =  32,561, d =    123
            "colon-cancer", "colon-cancer",                     # already scaled, n =   2,000, d =     62
            "leukemia_full", "leukemia_full"]                   # already scaled, n =      62, d =  7,129

scalings = ["column-scaling", "column-scaling",
            "column-scaling", "column-scaling",
            "column-scaling", "column-scaling",
            "column-scaling", "column-scaling",
            "none", "none",
            "none", "none",
            "none", "none",
            "none", "none"]

lambdas = [10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3)]



@time begin
@sync @distributed for idx_prob in problems
    data = datasets[idx_prob]
    scaling = scalings[idx_prob]
    lambda = lambdas[idx_prob]
    println("EXPERIMENT : ", idx_prob, " over ", length(problems))
    @printf "Inputs: %s + %s + %1.1e \n" data scaling lambda;

    Random.seed!(1)

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

    ## Computing theoretical optimal mini-batch size for b-nice sampling with inner loop size m = n
    b_theoretical = optimal_minibatch_Free_SVRG_nice(n, n, mu, L, Lmax) # optimal b for Free-SVRG when m=n

    ## Computing the empirical mini-batch size over a grid
    minibatchgrid = grids[idx_prob]
    println("---------------------------------- MINI-BATCH GRID ------------------------------------------")
    println(minibatchgrid)
    println("---------------------------------------------------------------------------------------------")

    OUTPUTS, itercomplex = simulate_Free_SVRG_nice(prob, minibatchgrid, options, numsimu=numsimu, skip_multiplier=skip_multipliers[idx_prob], path=save_path)

    ## Checking that all simulations reached tolerance
    fails = [OUTPUTS[i].fail for i=1:length(minibatchgrid)*numsimu]
    if all(s->(string(s)=="tol-reached"), fails)
        println("Tolerance always reached")
    else
        println("Some total complexities might be threshold because of reached maximal time")
        println("fails: ", fails)
    end

    ## Computing the empirical complexity
    empcomplex = reshape([OUTPUTS[i].epochs[end] for i=1:length(minibatchgrid)*numsimu], length(minibatchgrid)) # number of stochastic gradients computed
    min_empcomplex, idx_min = findmin(empcomplex)
    b_empirical = minibatchgrid[idx_min]

    ## Saving the result of the simulations
    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
    savename = string(probname, "-exp3-$(machine)-", numsimu, "-avg")
    savename = string(savename, "_skip_mult_", replace(string(skip_multipliers[idx_prob]), "." => "_")) # Extra suffix to check which skip values to keep
    # save("$(save_path)data/$(savename).jld",
    #      "options", options, "minibatchgrid", minibatchgrid,
    #      "itercomplex", itercomplex, "empcomplex", empcomplex,
    #      "b_theoretical", b_theoretical, "b_empirical", b_empirical)

    ## Plotting total complexity vs mini-batch size
    # legendpos = :topleft
    legendpos = :best
    pyplot()


end
end

println("\n\n--- EXPERIMENT 1.A FINISHED ---")