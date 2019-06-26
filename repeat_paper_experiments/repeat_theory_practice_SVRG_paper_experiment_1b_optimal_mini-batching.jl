"""
### "Towards closing the gap between the theory and practice of SVRG", Francis Bach, Othmane Sebbouh, Nidham Gazagnadou, Robert M. Gower (2019)

## --- EXPERIMENT 1.B ---
Goal: Compare SVRG variants: Bubeck version, Free-SVRG, Leap-SVRG and Loopless-SVRG-Decreasing for b-nice sampling. The mini-batch size is set to one for SVRG-Bubeck and to the optimal value b^*(n) corresponding to m=n.

## --- THINGS TO CHANGE BEFORE RUNNING ---
- line 37: enter your full path to the "StochOpt.jl/" repository in the *path* variable

## --- HOW TO RUN THE CODE ---
To run this experiment, open a terminal, go into the "StochOpt.jl/" repository and run the following command:
>julia -p <number_of_processor_to_add> repeat_paper_experiments/repeat_theory_practice_SVRG_paper_experiment_1b_optimal_mini-batching.jl <boolean>
where <number_of_processor_to_add> has to be replaced by the user.
- If <boolean> == false, only the first problem (ijcnn1_full + column-scaling + lambda=1e-1) is launched
- Else, <boolean> == true, all XX problems are launched

## --- EXAMPLE OF RUNNING TIME ---
Running time of the first problem only
XXXX, around 5min
Running time of all problems when adding XX processors on XXXX
XXXX, around XXmin

## --- SAVED FILES ---
For each problem (data set + scaling process + regularization)
- the empirical total complexity v.s. mini-batch size plots are saved in ".pdf" format in the "./experiments/theory_practice_SVRG/exp1b/figures/" folder
- the results of the simulations (mini-batch grid, empirical complexities, optimal empirical mini-batch size, etc.) are saved in ".jld" format in the "./experiments/theory_practice_SVRG/exp1b/outputs/" folder
"""

## General settings
max_epochs = 10^8
max_time = 60.0*60.0*24.0
precision = 10.0^(-6)

## File names
# details = "final"
# details = "legend"
details = "test-rho"

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

    pyplot() # No problem with pyplot when called in @everywhere statement
end

## Path settings
save_path = "$(path)experiments/theory_practice_SVRG/exp1b/"
#region
# Create saving directories if not existing
if !isdir("$(path)experiments/")
    mkdir("$(path)experiments/")
end
if !isdir("$(path)experiments/theory_practice_SVRG/")
    mkdir("$(path)experiments/theory_practice_SVRG/")
end
if !isdir(save_path)
    mkdir(save_path)
end
if !isdir("$(save_path)data/")
    mkdir("$(save_path)data/")
end
if !isdir("$(save_path)figures/")
    mkdir("$(save_path)figures/")
end
if !isdir("$(save_path)outputs/")
    mkdir("$(save_path)outputs/")
end
#endregion

## Experiments settings
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

## Set smaller number of skipped iteration for finer estimations (yet, longer simulations)
skip_errors = [[700 7000 -2. 7000],  # 1)  ijcnn1_full + scaled + 1e-1                 midnight retry / FINAL
               [13000 7000 -2. 5000],  # 2)  ijcnn1_full + scaled + 1e-3               midnight retry / FINAL
               [50000 30000 -2. 20000],  # 3)  YearPredictionMSD_full + scaled + 1e-1  midnight retry / FINAL
               [60000 15000 -2. 5000],  # 4)  YearPredictionMSD_full + scaled + 1e-3   midnight retry / FINAL
               [50000 2500 -2. 2500],  # 5)  slice + scaled + 1e-1                     100 epochs / FINAL
               [50000 1 -2. 1],  # 6)  slice + scaled + 1e-3                           100 epochs / FINAL
               [  10 2000 -2. 4000],  # 7)  real-sim + unscaled + 1e-1                 midnight retry / FINAL
               [500 5000 -2. 2000]]  # 8) real-sim + unscaled + 1e-3                   midnight retry / FINAL

@time @sync @distributed for idx_prob in problems
    data = datasets[idx_prob]
    scaling = scalings[idx_prob]
    lambda = lambdas[idx_prob]
    skip_error = skip_errors[idx_prob]
    println("EXPERIMENT : ", idx_prob, " over ", length(problems))
    @printf "Inputs: %s + %s + %1.1e \n" data scaling lambda

    Random.seed!(1)

    if idx_prob == 3
        global max_epochs = 10
    elseif idx_prob == 4
        global max_epochs = 13
    elseif idx_prob == 5 || idx_prob == 6
        global max_epochs = 100
    end

    ## Loading the data
    println("--- Loading data ---")
    data_path = "$(path)data/";
    X, y = loadDataset(data_path, data)

    ## Setting up the problem
    println("\n--- Setting up the selected problem ---")
    options = set_options(tol=precision, max_iter=10^8,
                          max_epocs=max_epochs,
                          max_time=max_time, # 60.0*60.0*10.0
                          skip_error_calculation=10^5,
                          batchsize=1,
                          regularizor_parameter="normalized",
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

    println("\nSTRONG CONVEXITY : ", prob.mu, "\n")

    X = nothing
    y = nothing

    n = prob.numdata
    mu = prob.mu
    Lmax = prob.Lmax
    L = prob.L

    ## Running methods
    OUTPUTS = [] # list of saved outputs

    ################################################################################
    ############################# OPTIMAL SVRG-BUBECK ##############################
    ################################################################################
    ## SVRG-Bubeck with 1-nice sampling ( m = m^*, b = 1, step size = gamma^* )
    numinneriters = -1                             # theoretical inner loop size (m^* = 20*Lmax/mu) set in initiate_SVRG_bubeck
    options.batchsize = 1                          # mini-batch size set to 1
    options.stepsize_multiplier = -1.0             # theoretical step size (gamma^* = 1/10*Lmax) set in boot_SVRG_bubeck
    sampling = build_sampling("nice", n, options)
    bubeck = initiate_SVRG_bubeck(prob, options, sampling, numinneriters=numinneriters)

    ## Setting the number of skipped iteration to m/4
    options.skip_error_calculation = skip_error[1] # skip error different for each algo
    # options.skip_error_calculation = round(Int64, bubeck.numinneriters/4)

    println("-------------------- WARM UP --------------------")
    tmp = options.max_epocs
    options.max_epocs = 3
    minimizeFunc(prob, bubeck, options)
    options.max_epocs = tmp
    bubeck.reset(prob, bubeck, options)
    println("-------------------------------------------------\n")

    out_bubeck = minimizeFunc(prob, bubeck, options)

    str_m_bubeck = @sprintf "%d" bubeck.numinneriters
    str_step_bubeck = @sprintf "%.2e" bubeck.stepsize
    # out_bubeck.name = latexstring("SVRG-Bubeck \$(m_{Bubeck}^* = $str_m_bubeck, b = 1, \\alpha_{Bubeck}^* = $str_step_bubeck)\$")
    out_bubeck.name = latexstring("SVRG \$(m^* = $str_m_bubeck, b = 1, \\alpha^* = $str_step_bubeck)\$")
    OUTPUTS = [OUTPUTS; out_bubeck]
    println("\n")

    ################################################################################
    ############################## OPTIMAL FREE-SVRG ###############################
    ################################################################################
    ## Free-SVRG with optimal b-nice sampling ( m = n, b = b^*(n), step size = gamma^*(b^*) )
    numinneriters = n                                                                         # inner loop size set to the number of data points
    # options.batchsize = optimal_minibatch_Free_SVRG_nice(numinneriters, n, mu, L, Lmax)     # mini-batch size set to the optimal value for m=n (same for Free- and Leap-SVRG)
    # Mini-batch size set to the optimal value for m=n taking sketch residual into acount
    # /!\ Not the same for Leap-SVRG anymore, but can be used for the latter as an approximation
    options.batchsize = optimal_minibatch_Free_SVRG_nice_tight(numinneriters, n, mu, L, Lmax)
    options.stepsize_multiplier = -1.0                                                        # theoretical step size set in boot_Free_SVRG
    sampling = build_sampling("nice", n, options)
    free = initiate_Free_SVRG(prob, options, sampling, numinneriters=numinneriters, averaged_reference_point=true)

    ## Setting the number of skipped iteration to m/4
    options.skip_error_calculation = skip_error[2] # skip error different for each algo
    # options.skip_error_calculation = round(Int64, free.numinneriters/4)

    out_free = minimizeFunc(prob, free, options)

    str_m_free = @sprintf "%d" free.numinneriters
    str_b_free = @sprintf "%d" free.batchsize
    str_step_free = @sprintf "%.2e" free.stepsize
    # out_free.name = latexstring("Free-SVRG \$(m = n = $str_m_free, b_{Free}^*(n) = $str_b_free, \\alpha_{Free}^*(b^*) = $str_step_free)\$")
    out_free.name = latexstring("Free-SVRG \$(m = n = $str_m_free, b^*(n) = $str_b_free, \\alpha^*(b^*) = $str_step_free)\$")
    OUTPUTS = [OUTPUTS; out_free]
    println("\n")

    #region
    ################################################################################
    ############################## OPTIMAL LEAP-SVRG ###############################
    ################################################################################
    # ## Leap-SVRG with optimal b-nice sampling ( p = 1/n, b = b^*(1/n), step sizes = {eta^*=1/L, alpha^*(b^*)} )
    # proba = 1/n                                                             # update probability set to the inverse of the number of data points
    # options.batchsize = optimal_minibatch_Free_SVRG_nice(n, n, mu, L, Lmax) # mini-batch size set to the optimal value for p=1/n (same for Free- and Leap-SVRG)
    # options.stepsize_multiplier = -1.0                                      # theoretical step sizes set in boot_Leap_SVRG
    # sampling = build_sampling("nice", n, options)
    # leap = initiate_Leap_SVRG(prob, options, sampling, proba)

    # ## Setting the number of skipped iteration to 1/4*p
    # options.skip_error_calculation = skip_error[3] # skip error different for each algo
    # # options.skip_error_calculation = round(Int64, 1/(4*proba))

    # out_leap = minimizeFunc(prob, leap, options)

    # str_proba_leap = @sprintf "%.2e" proba
    # str_b_leap = @sprintf "%d" leap.batchsize
    # str_step_sto_leap = @sprintf "%.2e" leap.stochastic_stepsize
    # str_step_grad_leap = @sprintf "%.2e" leap.gradient_stepsize
    # out_leap.name = latexstring("Leap-SVRG \$(p = 1/n = $str_proba_leap, b_{Leap}^*(n) = $str_b_leap, \\eta_{Leap}^* = $str_step_grad_leap, \\alpha_{Leap}^*(b^*) = $str_step_sto_leap)\$")
    # OUTPUTS = [OUTPUTS; out_leap]
    # println("\n")
    #endregion

    ################################################################################
    ############################### OPTIMAL L-SVRG-D ###############################
    ################################################################################
    ## L_SVRG_D with optimal b-nice sampling ( p = 1/n, b = b^*(1/n), step size = gamma^*(b^*) )
    proba = 1/n                                                                # update probability set to the inverse of the number of data points
    options.batchsize = optimal_minibatch_L_SVRG_D_nice(proba, n, mu, L, Lmax) # mini-batch size set to the optimal value for p=1/n
    options.stepsize_multiplier = -1.0                                         # theoretical step sizes set in boot_L_SVRG_D
    sampling = build_sampling("nice", n, options)
    decreasing = initiate_L_SVRG_D(prob, options, sampling, proba)

    ## Setting the number of skipped iteration to 1/4*p
    options.skip_error_calculation = skip_error[4] # skip error different for each algo
    # options.skip_error_calculation = round(Int64, 1/(4*proba))

    out_decreasing = minimizeFunc(prob, decreasing, options)

    str_proba_decreasing = @sprintf "%.2e" proba
    str_b_decreasing = @sprintf "%d" decreasing.batchsize
    str_step_decreasing = @sprintf "%.2e" decreasing.initial_stepsize
    # out_decreasing.name = latexstring("L-SVRG-D \$(p = 1/n = $str_proba_decreasing, b_{Decrease}^*(n) = $str_b_decreasing, \\alpha_{Decrease}^*(b^*) = $str_step_decreasing)\$")
    out_decreasing.name = latexstring("L-SVRG-D \$(p = 1/n = $str_proba_decreasing, b^*(n) = $str_b_decreasing, \\alpha^*(b^*) = $str_step_decreasing)\$")
    OUTPUTS = [OUTPUTS; out_decreasing]
    println("\n")

    ## Saving outputs and plots
    if path == "/home/infres/ngazagnadou/StochOpt.jl/"
        suffix = "lame23"
    end
    savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
    savename = string(savename, "-exp1b-$(suffix)-$(details)")
    save("$(save_path)outputs/$(savename).jld", "OUTPUTS", OUTPUTS)

    if idx_prob == 8
        legendpos = :best
    else
        legendpos = :topright
    end

    pyplot()
    # plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp1b-$(suffix)-$(max_epochs)_max_epochs", path=save_path, legendpos=legendpos, legendfont=6) # Plot and save output
    plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp1b-$(suffix)-$(details)", path=save_path, legendpos=legendpos, legendfont=8)

end
println("\n\n--- EXPERIMENT 1.B FINISHED ---")