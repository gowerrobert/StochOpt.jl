"""
### "Towards closing the gap between the theory and practice of SVRG", O. Sebbouh, S. Jelassi, N. Gazagnadou, F. Bach, R. M. Gower (2019)

## --- EXPERIMENT 2.A ---
Goal: Comparing Free-SVRG with m=n for different mini-batch sizes {1, 100, \\sqrt{n}, n, b^*}.

## --- THINGS TO CHANGE BEFORE RUNNING ---


## --- HOW TO RUN THE CODE ---
To run this experiment, open a terminal, go into the "StochOpt.jl/" repository and run the following command:
>julia repeat_paper_experiments/repeat_theory_practice_SVRG_paper_experiment_2a_free_minibatch.jl

## --- EXAMPLE OF RUNNING TIME ---
5min for false

## --- SAVED FILES ---

"""

## General settings
max_epochs = 10^8
max_time = 60.0*60.0*24.0
precision = 10.0^(-6)

## File names
details = "final"
# details = "test-rho"
# details = "legend"

## Bash input
all_problems = parse(Bool, ARGS[1]) # run 1 (false) or all the 8 problems (true)

using Distributed

@everywhere begin
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

## Create saving directories if not existing
save_path = "$(path)experiments/theory_practice_SVRG/"
#region
if !isdir(save_path)
    mkdir(save_path)
end
save_path = "$(save_path)exp2a/"
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
if all_problems
    problems = 1:8
else
    problems = 1:1
end

datasets = ["ijcnn1_full", "ijcnn1_full",                       # scaled,   n = 141,691, d =     22
            "YearPredictionMSD_full", "YearPredictionMSD_full", # scaled,   n = 515,345, d =     90
            "slice", "slice",                                   # scaled,   n =  53,500, d =    384
            "real-sim", "real-sim"]                             # unscaled, n =  72,309, d = 20,958

scalings = ["column-scaling", "column-scaling",
            "column-scaling", "column-scaling",
            "column-scaling", "column-scaling",
            "none", "none"]

lambdas = [10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3)]

## Set smaller number of skipped iteration for more data points
skip_errors = [[7000 5000 3000 3 7000],      # 1) ijcnn1_full + scaled + 1e-1              b^* = 1
               [7000 5000 3000 70 7000],     # 2) ijcnn1_full + scaled + 1e-3              b^* = 1
               [30000 20000 10000 10 30000], # 3) YearPredictionMSD_full + scaled + 1e-1   b^* = 1          less than 3h
               [30000 20000 10000 10 30000], # 4) YearPredictionMSD_full + scaled + 1e-3   b^* = 2
               [25000 2000 1000 1 2500],     # 5) slice + scaled + 1e-1                    b^* = 22         less than 3h
               [25000 2000 1000 1 2500],     # 6) slice + scaled + 1e-3                    b^* = n = 53500  less than 3h
               [2000 2000 1000 1 2000],      # 7) real-sim + unscaled + 1e-1               b^* = 1
               [5000 5000 2000 1 8000]]      # 8) real-sim + unscaled + 1e-3               b^* = 1

@time begin
@sync @distributed for idx_prob in problems
    data = datasets[idx_prob]
    scaling = scalings[idx_prob]
    lambda = lambdas[idx_prob]
    skip_error = skip_errors[idx_prob]
    println("EXPERIMENT : ", idx_prob, " over ", length(problems))
    @printf "Inputs: %s + %s + %1.1e \n" data scaling lambda

    Random.seed!(1)

    if idx_prob == 5 || idx_prob == 6
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
                        #   max_epocs=max_epochs_list[idx_prob], #limiting computation time
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
    # b_theoretical = optimal_minibatch_Free_SVRG_nice(n, n, mu, L, Lmax) # optimal b for Free-SVRG when m=n
    # Mini-batch size set to the optimal value for m=n taking sketch residual into acount
    # /!\ Not the same for Leap-SVRG anymore, but can be used for the latter as an approximation
    b_theoretical = optimal_minibatch_Free_SVRG_nice_tight(n, n, mu, L, Lmax)
    println("------------------------------------------------------------")
    println("Theoretical mini-batch: ", b_theoretical)
    println("------------------------------------------------------------\n")

    ## List of mini-batch sizes
    minibatch_list   = [1, 100, round(Int64, sqrt(n)), n]
    minibatch_labels = ["", "", " \\sqrt{n} =", " n ="]
    if b_theoretical == 1
        minibatch_labels[1] = " b^*(n) ="
    elseif b_theoretical == n
        minibatch_labels[4] = " b^*(n) = n ="
    elseif !(b_theoretical in minibatch_list) # low proba that b^* is 100 or sqrt(n)
        minibatch_list   = [minibatch_list   ; b_theoretical]
        minibatch_labels = [minibatch_labels ; " b^*(n) ="]
    end

    ## Running methods
    OUTPUTS = [] # list of saved outputs

    ## Launching Free-SVRG for different mini-batch sizes and m = n
    for idx_minibatch in 1:length(minibatch_list)
        ## Monitoring
        minibatch_label = minibatch_labels[idx_minibatch]
        str_minibatch = @sprintf "%d" minibatch_list[idx_minibatch]
        println("\n------------------------------------------------------------")
        println("Current mini-batch: \$b =$minibatch_label $str_minibatch\$")
        println("------------------------------------------------------------")

        numinneriters = n                                  # inner loop size set to the number of data points
        options.batchsize = minibatch_list[idx_minibatch]  # mini-batch size
        options.stepsize_multiplier = -1.0                 # theoretical step size set in boot_Free_SVRG
        sampling = build_sampling("nice", n, options)
        free = initiate_Free_SVRG(prob, options, sampling, numinneriters=numinneriters, averaged_reference_point=true)

        ## Setting the number of skipped iteration
        options.skip_error_calculation = skip_error[idx_minibatch] # skip error different for each mini-batch size

        ## Running the minimization
        output = minimizeFunc(prob, free, options)

        str_step = @sprintf "%.2e" free.stepsize
        output.name = latexstring("\$b =$minibatch_label $str_minibatch, \\alpha^*(b) = $str_step\$")
        OUTPUTS = [OUTPUTS; output]
        println("\n")
    end
    println("\n")

    ## Saving outputs and plots
    if path == "/home/infres/ngazagnadou/StochOpt.jl/"
        suffix = "lame23"
    else
        suffix = ""
    end
    savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
    savename = string(savename, "-exp2a-$(suffix)-$(details)")
    save("$(save_path)outputs/$(savename).jld", "OUTPUTS", OUTPUTS)

    if idx_prob == 5 || idx_prob == 6
        legendpos = :best
    else
        legendpos = :topright
    end

    legendtitle = "Mini-batch size b"
    pyplot()
    # plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp2a-$(suffix)-$(details)", path=save_path, legendpos=legendpos, legendfont=8)
    plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp2a-$(suffix)-$(details)", path=save_path, legendpos=legendpos, legendtitle=legendtitle, legendfont=8)

end
end

println("\n\n--- EXPERIMENT 2.A FINISHED ---")