"""
### "Towards closing the gap between the theory and practice of SVRG", O. Sebbouh, S. Jelassi, N. Gazagnadou, F. Bach, R. M. Gower (2019)

## --- EXPERIMENT 4 ---
Goal: Comparing Free-SVRG for different inner loop sizes {n, L_max/mu, m^* = L_max/mu, 2n} for 1-nice sampling.

## --- THINGS TO CHANGE BEFORE RUNNING ---


## --- HOW TO RUN THE CODE ---
To run this experiment, open a terminal, go into the "StochOpt.jl/" repository and run the following command:
>julia repeat_paper_experiments/repeat_theory_practice_SVRG_paper_experiment_4_vs_m.jl

## --- EXAMPLE OF RUNNING TIME ---

## --- SAVED FILES ---

"""

## General settings
max_epochs = 10^8
max_time = 60.0*60.0 #60.0*60.0*4.0
precision = 10.0^(-6)

## File names
# details = "final"
details = "1h"
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
save_path = "$(save_path)exp4/"
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
#          m =   n      2n    Lmax/mu   m*
skip_errors = [[7000   7000     200    200],             # 1)  ijcnn1_full + scaled + 1e-1              m^* =
               [7000   7000    6500   6500],             # 2)  ijcnn1_full + scaled + 1e-3              m^* =
               [30000  30000   40000  40000],    # 3)  YearPredictionMSD_full + scaled + 1e-1           m^* =
               [40000  40000   10000  10000],    # 4)  YearPredictionMSD_full + scaled + 1e-3           m^* =
               [-2      -2     -2      -2],                  # 5)  covtype_binary + scaled + 1e-1
               [-2      -2     -2      -2],                  # 6)  covtype_binary + scaled + 1e-3
               [40000  40000   50000  50000],              # 7)  slice + scaled + 1e-1                  m^* =
               [40000  40000   50000  50000],              # 8)  slice + scaled + 1e-3                  m^* =
               [2000   2000      3      3],              # 9)  real-sim + unscaled + 1e-1               m^* =
               [5000   5000     150    150],              # 10) real-sim + unscaled + 1e-3              m^* =
               [-2 -2 -2 -2 -2],                  # 11) a1a_full + unscaled + 1e-1
               [-2 -2 -2 -2 -2],                  # 12) a1a_full + unscaled + 1e-3
               [-2 -2 -2 -2 -2],                  # 13) colon-cancer + unscaled + 1e-1
               [-2 -2 -2 -2 -2],                  # 14) colon-cancer + unscaled + 1e-3
               [-2 -2 -2 -2 -2],                  # 15) leukemia_full + unscaled + 1e-1
               [-2 -2 -2 -2 -2]]                  # 16) leukemia_full + unscaled + 1e-3

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

    m_star = round(Int64, (3*Lmax)/mu) # theoretical optimal inner loop size for Free-SVRG with 1-nice sampling

    ## List of mini-batch sizes
    numinneriters_list   = [n, 2*n, round(Int64, Lmax/mu), m_star]
    numinneriters_labels = ["n", "2n", "L_{\\max}/\\mu", "3L_{\\max}/\\mu = m^*"] # round? floor?

    ## Running methods
    OUTPUTS = [] # list of saved outputs

    ## Launching Free-SVRG for different mini-batch sizes and m = n
    for idx_numinneriter in 1:length(numinneriters_list)
        ## Monitoring
        numinneriters_label = numinneriters_labels[idx_numinneriter]
        str_numinneriters = @sprintf "%d" numinneriters_list[idx_numinneriter]
        println("\n------------------------------------------------------------")
        println("Current inner loop size: \$m = $numinneriters_label = $str_numinneriters\$")
        println("------------------------------------------------------------")

        numinneriters = numinneriters_list[idx_numinneriter]  # inner loop size set to the number of data points
        options.batchsize = 1                                 # mini-batch size set to 1
        options.stepsize_multiplier = -1.0                    # theoretical step size set in boot_Free_SVRG
        sampling = build_sampling("nice", n, options)
        free = initiate_Free_SVRG(prob, options, sampling, numinneriters=numinneriters, averaged_reference_point=true)

        ## Setting the number of skipped iteration
        options.skip_error_calculation = skip_error[idx_numinneriter] # skip error different for each mini-batch size

        ## Running the minimization
        output = minimizeFunc(prob, free, options)

        output.name = latexstring("\$$numinneriters_label = $str_numinneriters\$")
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
    savename = string(savename, "-exp4-$(suffix)-$(details)")
    save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

    legendpos = :topright
    legendtitle = "Inner loop size m"
    pyplot()
    # plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp4-$(suffix)-$(details)", path=save_path, legendpos=legendpos, legendfont=8)
    plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp4-$(suffix)-$(details)", path=save_path, legendpos=legendpos, legendtitle=legendtitle, legendfont=8)

end
end

println("\n\n--- EXPERIMENT 4 FINISHED ---")