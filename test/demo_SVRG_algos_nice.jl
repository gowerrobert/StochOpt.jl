using Distributed

@everywhere begin
    path = "/home/nidham/phd/StochOpt.jl/" # Change the full path here
    # path = "/cal/homes/ngazagnadou/StochOpt.jl/"

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

# using JLD
# using Plots
# using StatsBase
# using Match
# using Combinatorics
# using Random
# using Printf
# using LinearAlgebra
# using Statistics
# using Base64
# include("../src/StochOpt.jl")

## Path settings
#region
save_path = "./experiments/SVRG/"
if !isdir(save_path) # create directory if not existing
    if !isdir("./experiments/")
        mkdir("./experiments/")
    end
    mkdir(save_path)
    mkdir("$(save_path)data/")
    mkdir("$(save_path)figures/")
end

if !isdir("$(save_path)data/")
    mkdir("$(save_path)data/")
end

if !isdir("$(save_path)figures/")
    mkdir("$(save_path)figures/")
end
#endregion

# Random.seed!(1)

# lambda = 1e-3
# scaling = "column-scaling"

## Basic parameters and options for solvers
# options = set_options(max_iter=10^8, max_time=10.0^5, max_epocs=20, initial_point="zeros", skip_error_calculation = 1000)
# options = set_options(max_iter=10^8, max_time=10.0^5, max_epocs=10, initial_point="zeros", skip_error_calculation = 100)
# options = set_options(max_iter=10^8, max_time=10.0^5, max_epocs=15, initial_point="zeros", skip_error_calculation = 1)

## Debugging settings
# options = set_options(max_iter=20, max_time=10.0^5, max_epocs=10^5, initial_point="zeros", skip_error_calculation=1)
# numinneriters = 5
# proba = 1/numinneriters

## Load problem
# datapath = "./data/"
# data = "australian"
# data = "ijcnn1_full"    # n = 141,691, d = 23
# data = "YearPredictionMSD_full"


datasets = ["australian", "australian",                             # scaled,   n =     690, d =     15
            "slice", "slice",                                       # scaled,   n =  53,500, d =    384
            "ijcnn1_full", "ijcnn1_full",                           # scaled,   n = 141,691, d =     22
            "YearPredictionMSD_full", "YearPredictionMSD_full"]     # scaled,   n = 515,345, d =     90

lambdas = [10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3),
           10^(-1), 10^(-3)]

skip_error = [200,           # XXmin with XXX
              200,           # XXmin with XXX
              10000,         # XXmin with XXX
              10000,         # XXmin with XXX
              50000,         # XXmin with XXX
              100000,        # XXmin with XXX
              100000,        # XXmin with XXX
              100000]        # XXmin with XXX


problems = 4:8
# problems = 1:2

@sync @distributed for idx_prob in problems
    data = datasets[idx_prob]
    scaling = "column-scaling"
    lambda = lambdas[idx_prob]
    println("EXPERIMENT : ", idx_prob, " over ", length(problems))
    @printf "Inputs: %s + %s + %1.1e \n" data scaling lambda

    Random.seed!(1)

    ## Loading the data
    println("--- Loading data ---")
    data_path = "$(path)data/";
    X, y = loadDataset(data_path, data)

    ## Setting up the problem
    println("\n--- Setting up the selected problem ---")
    skip_parameter = skip_error[idx_prob]
    options = set_options(max_iter=10^8, max_time=10.0^5, max_epocs=10, initial_point="zeros", skip_error_calculation=skip_parameter)

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

    ## Running methods
    OUTPUTS = [] # list of saved outputs

    ################################################################################
    ########################## DOUBLE-LOOP SVRG VARIANTS ###########################
    ################################################################################

    ## SVRG-Bubeck with b-nice sampling (m = m^*, b = 1, step size = gamma^*)
    options.stepsize_multiplier = -1.0 # theoretical step sizes in boot_Leap_SVRG
    options.batchsize = 1
    sampling = build_sampling("nice", prob.numdata, options)
    options.stepsize_multiplier = -1.0 # Theoretical step size in boot_SVRG_bubeck
    numinneriters = -1 # 20*Lmax/mu
    bubeck = initiate_SVRG_bubeck(prob, options, sampling, numinneriters=numinneriters)

    println("-------------------- WARM UP --------------------")
    tmp = options.max_epocs
    options.max_epocs = 3
    minimizeFunc(prob, bubeck, options) # Warm up
    options.max_epocs = tmp
    bubeck.reset(prob, bubeck, options)
    println("-------------------------------------------------")

    output2 = minimizeFunc(prob, bubeck, options)
    str_m_2 = @sprintf "%d" bubeck.numinneriters
    str_b_2 = @sprintf "%d" bubeck.batchsize
    str_step_2 = @sprintf "%.2e" bubeck.stepsize
    output2.name = latexstring("$(output2.name) \$(m^* = $str_m_2, b = $str_b_2 , \\gamma^*(b) = $str_step_2)\$") # if we want to set m^* and b > 1 in Bubeck SVRG
    # output2.name = latexstring("$(output2.name) \$(m = $str_m_2, b = $str_b_2, \\gamma^*(b) = $str_step_2)\$")
    OUTPUTS = [OUTPUTS; output2]


    ## Free-SVRG with b-nice sampling
    options.stepsize_multiplier = -1.0 # theoretical step sizes in boot_Leap_SVRG

    ## m = n, b = 1, step size = gamma^*(b)
    numinneriters = prob.numdata
    options.batchsize = 1

    ## m = n, b = b^*(n), step size = gamma^*(b^*(n))
    # numinneriters = prob.numdata
    # if numinneriters == prob.numdata
    #     options.batchsize = optimal_minibatch_Free_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)
    # else
    #     options.batchsize = 1 # default value for other inner loop sizes
    # end

    sampling = build_sampling("nice", prob.numdata, options)
    free = initiate_Free_SVRG(prob, options, sampling, numinneriters=numinneriters, averaged_reference_point=true)
    output3 = minimizeFunc(prob, free, options)
    str_m_3 = @sprintf "%d" free.numinneriters
    str_b_3 = @sprintf "%d" free.batchsize
    str_step_3 = @sprintf "%.2e" free.stepsize
    output3.name = latexstring("$(output3.name) \$(m = n = $str_m_3, b^*(n) = $str_b_3 , \\gamma^*(b^*(n)) = $str_step_3)\$") # optimal b^*
    # output3.name = latexstring("$(output3.name) \$(m = $str_m_3, b = $str_b_3, \\gamma^*(b) = $str_step_3)\$")
    OUTPUTS = [OUTPUTS; output3]


    ################################################################################
    ############################ LOOPLESS SVRG VARIANTS ############################
    ################################################################################

    ## Leap-SVRG with b-nice sampling
    options.stepsize_multiplier = -1.0 # theoretical step sizes in boot_Leap_SVRG

    ## p = 1/n, b = 1, step sizes = {eta^*, alpha^*(b)}
    proba = 1/prob.numdata
    options.batchsize = 1

    ## p = 1/n, b = b^*(1/n), step sizes = {eta^*, alpha^*(b^*(1/n))}
    # proba = 1/prob.numdata
    # if abs(proba - 1/prob.numdata) < 1e-7
    #     options.batchsize = optimal_minibatch_Free_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)
    # else
    #     options.batchsize = 1 # default value for other inner loop sizes
    # end

    sampling = build_sampling("nice", prob.numdata, options)
    leap = initiate_Leap_SVRG(prob, options, sampling, proba)
    output4 = minimizeFunc(prob, leap, options)
    str_b_4 = @sprintf "%d" sampling.batchsize
    str_proba_4 = @sprintf "%.2e" proba
    str_step_sto_4 = @sprintf "%.2e" leap.stochastic_stepsize
    str_step_grad_4 = @sprintf "%.2e" leap.gradient_stepsize
    output4.name = latexstring("$(output4.name) \$(p = 1/n = $str_proba_4, b^*(n) = $str_b_4, \\eta^* = $str_step_grad_4, \\alpha^*(b^*(n)) = $str_step_sto_4)\$") # optimal b^*
    # output4.name = latexstring("$(output4.name) \$(p = $str_proba_4, b = $str_b_4, \\eta^* = $str_step_grad_4, \\alpha^*(b) = $str_step_sto_4)\$")
    OUTPUTS = [OUTPUTS; output4]


    ## L_SVRG_D with b-nice sampling
    options.stepsize_multiplier = -1.0 # theoretical step size in boot_L_SVRG_D

    ## p = 1/n, b = 1, step size = gamma^*(b)
    proba = 1/prob.numdata
    options.batchsize = 1

    ## p = 1/n, b = b^*(1/n), step size = gamma^*(b^*(1/n))
    # proba = 1/prob.numdata
    # if abs(proba - 1/prob.numdata) < 1e-7
    #     options.batchsize = optimal_minibatch_Free_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)
    # else
    #     options.batchsize = 1 # default value for other inner loop sizes
    # end

    sampling = build_sampling("nice", prob.numdata, options)
    decreasing = initiate_L_SVRG_D(prob, options, sampling, proba)
    output5 = minimizeFunc(prob, decreasing, options)
    str_b_5 = @sprintf "%d" sampling.batchsize
    str_proba_5 = @sprintf "%.2e" proba
    str_step_5 = @sprintf "%.2e" decreasing.stepsize
    output5.name = latexstring("$(output5.name) \$(p = 1/n = $str_proba_5, b^*(n) = $str_b_5, \\gamma^*(b^*(n)) = $str_step_5)\$") # optimal b^*
    # output5.name = latexstring("$(output5.name) \$(p = $str_proba_5, b = $str_b_5, \\gamma^*(b) = $str_step_5)\$")
    OUTPUTS = [OUTPUTS; output5]

    ## Saving outputs and plots
    # savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
    # savename = string(savename, "-", "demo_SVRG_algos_nice")
    # save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

    pyplot() # gr() pyplot() # pgfplots() #plotly()
    plot_outputs_Plots(OUTPUTS, prob, options, methodname="comparison_SVRG_algos_nice", path=save_path, legendfont=8) # Plot and save output

end

# X, y = loadDataset(datapath, data)

# u = unique(y)
# if length(u) < 2
#     error("Wrong number of possible outputs")
# elseif length(u) == 2
#     println("Binary output detected: the problem is set to logistic regression")
#     prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling=scaling)
# else
#     println("More than three modalities in the outputs: the problem is set to ridge regression")
#     prob = load_ridge_regression(X, y, data, options, lambda=lambda, scaling=scaling)
# end

# X = nothing
# y = nothing


# ## Running methods
# OUTPUTS = [] # list of saved outputs

# ################################################################################
# ########################## DOUBLE-LOOP SVRG VARIANTS ###########################
# ################################################################################

# ## SVRG-Bubeck with b-nice sampling (m = m^*, b = 1, step size = gamma^*)
# options.stepsize_multiplier = -1.0 # theoretical step sizes in boot_Leap_SVRG
# options.batchsize = 1
# sampling = build_sampling("nice", prob.numdata, options)
# options.stepsize_multiplier = -1.0 # Theoretical step size in boot_SVRG_bubeck
# # numinneriters = -1 # 20*Lmax/mu
# bubeck = initiate_SVRG_bubeck(prob, options, sampling, numinneriters=numinneriters)

# # println("-------------------- WARM UP --------------------")
# # tmp = options.max_epocs
# # options.max_epocs = 3
# # minimizeFunc(prob, bubeck, options) # Warm up
# # options.max_epocs = tmp
# # bubeck.reset(prob, bubeck, options)
# # println("-------------------------------------------------")

# output2 = minimizeFunc(prob, bubeck, options)
# str_m_2 = @sprintf "%d" bubeck.numinneriters
# str_b_2 = @sprintf "%d" bubeck.batchsize
# str_step_2 = @sprintf "%.2e" bubeck.stepsize
# # output2.name = latexstring("$(output2.name) \$(m^* = $str_m_2, b = $str_b_2 , \\gamma^*(b) = $str_step_2)\$") # if we want to set b > 1 in Bubeck SVRG
# output2.name = latexstring("$(output2.name) \$(m^* = $str_m_2, b = $str_b_2, \\gamma^*(b) = $str_step_2)\$")
# OUTPUTS = [OUTPUTS; output2]

# ## Free-SVRG with b-nice sampling (m = n, b = b^*(n), step size = gamma^*(b^*(n)))
# options.stepsize_multiplier = -1.0 # theoretical step sizes in boot_Leap_SVRG
# # numinneriters = prob.numdata
# # if numinneriters == prob.numdata
# #     options.batchsize = optimal_minibatch_Free_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)
# # else
# #     options.batchsize = 1 # default value for other inner loop sizes
# # end
# options.batchsize = 1
# sampling = build_sampling("nice", prob.numdata, options)
# free = initiate_Free_SVRG(prob, options, sampling, numinneriters=numinneriters, averaged_reference_point=true)
# output3 = minimizeFunc(prob, free, options)
# str_m_3 = @sprintf "%d" free.numinneriters
# str_b_3 = @sprintf "%d" free.batchsize
# str_step_3 = @sprintf "%.2e" free.stepsize
# # output3.name = latexstring("$(output3.name) \$(m = n = $str_m_3, b^*(n) = $str_b_3 , \\gamma^*(b^*(n)) = $str_step_3)\$") # optimal b^*
# output3.name = latexstring("$(output3.name) \$(m = $str_m_3, b = $str_b_3, \\gamma^*(b) = $str_step_3)\$")
# OUTPUTS = [OUTPUTS; output3]


# ################################################################################
# ############################ LOOPLESS SVRG VARIANTS ############################
# ################################################################################

# ## Leap-SVRG with b-nice sampling (m = n, b = b^*(n), step size = gamma^*(b^*(n)))
# options.stepsize_multiplier = -1.0 # theoretical step sizes in boot_Leap_SVRG
# # if numinneriters == prob.numdata
# #     options.batchsize = optimal_minibatch_Leap_SVRG_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)
# # else
# #     options.batchsize = 1 # default value for other inner loop sizes
# # end
# options.batchsize = 1
# sampling = build_sampling("nice", prob.numdata, options)
# # proba = 1/prob.numdata
# leap = initiate_Leap_SVRG(prob, options, sampling, proba)
# output4 = minimizeFunc(prob, leap, options)
# str_b_4 = @sprintf "%d" sampling.batchsize
# str_proba_4 = @sprintf "%.2e" proba
# str_step_sto_4 = @sprintf "%.2e" leap.stochastic_stepsize
# str_step_grad_4 = @sprintf "%.2e" leap.gradient_stepsize
# # output4.name = latexstring("$(output4.name) \$(p = 1/n = $str_proba_4, b^*(n) = $str_b_4, \\eta^* = $str_step_grad_4, \\alpha^*(b^*(n)) = $str_step_sto_4)\$") # optimal b^*
# output4.name = latexstring("$(output4.name) \$(p = $str_proba_4, b = $str_b_4, \\eta^* = $str_step_grad_4, \\alpha^*(b) = $str_step_sto_4)\$")
# OUTPUTS = [OUTPUTS; output4]

# ## L_SVRG_D with b-nice sampling (b = 1, proba=1/n, step sizes = gamma^*)
# options.stepsize_multiplier = -1.0 # theoretical step sizes in boot_Leap_SVRG
# # if numinneriters == prob.numdata
# #     options.batchsize = optimal_minibatch_L_SVRG_D_nice(prob.numdata, prob.mu, prob.L, prob.Lmax)
# # else
# #     options.batchsize = 1 # default value for other inner loop sizes
# # end
# options.batchsize = 1
# sampling = build_sampling("nice", prob.numdata, options)
# # proba = 1/prob.numdata
# decreasing = initiate_L_SVRG_D(prob, options, sampling, proba)
# output5 = minimizeFunc(prob, decreasing, options)
# str_b_5 = @sprintf "%d" sampling.batchsize
# str_proba_5 = @sprintf "%.3f" proba
# str_step_5 = @sprintf "%.2e" decreasing.stepsize
# # output5.name = latexstring("$(output5.name) \$(p = 1/n = $str_proba_5, b^*(n) = $str_b_5, \\gamma^*(b^*(n)) = $str_step_5)\$") # optimal b^*
# output5.name = latexstring("$(output5.name) \$(p = $str_proba_5, b = $str_b_5, \\gamma^*(b) = $str_step_5)\$")
# OUTPUTS = [OUTPUTS; output5]

# ## Saving outputs and plots
# # savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
# # savename = string(savename, "-", "demo_SVRG_algos_nice")
# # save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

# pyplot() # gr() pyplot() # pgfplots() #plotly()
# plot_outputs_Plots(OUTPUTS, prob, options, methodname="comparison_SVRG_algos_nice", path=save_path, legendfont=8) # Plot and save output





#region
# ## Vanilla-SVRG with 1-nice sampling (m = n, b = 1, step size = gamma^*)
# options.batchsize = 1
# sampling = build_sampling("nice", prob.numdata, options)
# options.stepsize_multiplier = -1.0 # 1/10Lmax
# numinneriters = prob.numdata # n
# SVRG_vanilla = initiate_SVRG_vanilla(prob, options, sampling, numinneriters=numinneriters)

# println("-------------------- WARM UP --------------------")
# # options.max_epocs = 10
# minimizeFunc(prob, SVRG_vanilla, options) # Warm up
# # options.max_epocs = 100
# SVRG_vanilla.reset(prob, SVRG_vanilla, options)
# println("-------------------------------------------------")

# output1 = minimizeFunc(prob, SVRG_vanilla, options)
# str_m_1 = @sprintf "%d" SVRG_vanilla.numinneriters
# str_b_1 = @sprintf "%d" SVRG_vanilla.batchsize
# str_step_1 = @sprintf "%.2e" SVRG_vanilla.stepsize
# output1.name = latexstring("$(output1.name) \$(m = 2n = $str_m_1, b = $str_b_1 , \\gamma^* = $str_step_1\$)")
# OUTPUTS = [OUTPUTS; output1]
#endregion
