"""
### "Towards closing the gap between the theory and practice of SVRG", O. Sebbouh, S. Jelassi, N. Gazagnadou, F. Bach, R. M. Gower (2019)

## --- EXPERIMENT 3 ---
Goal: Comparing Free-SVRG with m=n for different mini-batch sizes {1, 100, \\sqrt{n}, n, b^*}.

## --- THINGS TO CHANGE BEFORE RUNNING ---


## --- HOW TO RUN THE CODE ---
To run this experiment, open a terminal, go into the "StochOpt.jl/" repository and run the following command:
>julia repeat_paper_experiments/repeat_theory_practice_SVRG_paper_experiment_3_free_minibatch.jl

## --- EXAMPLE OF RUNNING TIME ---

## --- SAVED FILES ---

"""

## General settings
max_epochs = 10^8
max_time = 60.0*60.0*6.0 # 5000.0
precision = 10.0^(-6)

## File names
# details = "final"
details = "100_epochs"
# details = "test"

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

## Set smaller number of skipped iteration for more data points
skip_errors = [[7000 5000 3000 3 7000],             # 1)  ijcnn1_full + scaled + 1e-1              b^* = 1
               [7000 5000 3000 70 7000],             # 2)  ijcnn1_full + scaled + 1e-3              b^* = 1
               [30000 10000 5000 10 30000],    # 3)  YearPredictionMSD_full + scaled + 1e-1   b^* = 1                   moins de 3h a tourner
               [30000 10000 5000 1 30000],    # 4)  YearPredictionMSD_full + scaled + 1e-3   b^* = 2
               [-2 -2 -2 -2 -2],                  # 5)  covtype_binary + scaled + 1e-1
               [-2 -2 -2 -2 -2],                  # 6)  covtype_binary + scaled + 1e-3
               [25000 2000 1000 1 2500],              # 7)  slice + scaled + 1e-1                    b^* = 22            moins de 3h a tourner
               [25000 2000 1000 1 2500],              # 8)  slice + scaled + 1e-3                    b^* = n = 53500     moins de 3h a tourner
               [2000 1000 500 1 2000],              # 9)  real-sim + unscaled + 1e-1               b^* = 1
               [5000 2500 1000 1 5000],              # 10) real-sim + unscaled + 1e-3               b^* = 1 (for approx mu)
               [-2 -2 -2 -2 -2],                  # 11) a1a_full + unscaled + 1e-1
               [-2 -2 -2 -2 -2],                  # 12) a1a_full + unscaled + 1e-3
               [-2 -2 -2 -2 -2],                  # 13) colon-cancer + unscaled + 1e-1
               [-2 -2 -2 -2 -2],                  # 14) colon-cancer + unscaled + 1e-3
               [-2 -2 -2 -2 -2],                  # 15) leukemia_full + unscaled + 1e-1
               [-2 -2 -2 -2 -2]]                  # 16) leukemia_full + unscaled + 1e-3

# max_epochs_list = [ 300, # 1)  ijcnn1_full + scaled + 1e-1
#                    1000, # 2)  ijcnn1_full + scaled + 1e-3
#                    9999 , # 3)  YearPredictionMSD_full + scaled + 1e-1
#                    9999, # 4)  YearPredictionMSD_full + scaled + 1e-3
#                      -2,
#                      -2,
#                    9999, # 7)  slice + scaled + 1e-1
#                    9999, # 8)  slice + scaled + 1e-3
#                     250, # 9)  real-sim + unscaled + 1e-1
#                     500, # 10) real-sim + unscaled + 1e-3
#                      -2,
#                      -2,
#                      -2,
#                      -2,
#                      -2,
#                      -2]

@time begin
@sync @distributed for idx_prob in problems
    data = datasets[idx_prob]
    scaling = scalings[idx_prob]
    lambda = lambdas[idx_prob]
    skip_error = skip_errors[idx_prob]
    println("EXPERIMENT : ", idx_prob, " over ", length(problems))
    @printf "Inputs: %s + %s + %1.1e \n" data scaling lambda

    Random.seed!(1)

    if idx_prob == 7 || idx_prob == 8
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
    b_theoretical = optimal_minibatch_Free_SVRG_nice(n, n, mu, L, Lmax) # optimal b for Free-SVRG when m=n
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
    if path == "/cal/homes/ngazagnadou/StochOpt.jl/"
        suffix = "lame10"
    elseif path == "/home/infres/ngazagnadou/StochOpt.jl/"
        suffix = "lame23"
    else
        suffix = "home"
    end

    savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
    savename = string(savename, "-exp3-$(suffix)-$(details)")
    save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

    if idx_prob == 7 || idx_prob == 8
        legendpos = :best
    else
        legendpos = :topright
    end

    legendtitle = "Mini-batch size b"
    pyplot()
    # plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp3-$(suffix)-$(details)", path=save_path, legendpos=legendpos, legendfont=8)
    plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp3-$(suffix)-$(details)", path=save_path, legendpos=legendpos, legendtitle=legendtitle, legendfont=8)

end
end

println("\n\n--- EXPERIMENT 3 FINISHED ---")