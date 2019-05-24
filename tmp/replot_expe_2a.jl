## Detail in file names
details = "exact_mu"
problems = [1 2 3 4 7 8 9 10]

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

## Bash input
machine = "home" # ARGS[1]
if machine == "lame10"
    path = "/cal/homes/ngazagnadou/StochOpt.jl/"   # lame10
elseif machine == "lame23"
    path = "/home/infres/ngazagnadou/StochOpt.jl/" # lame23
elseif machine == "home"
    path = "/home/nidham/phd/StochOpt.jl/"         # local
end
println("path: ", path)

include("$(path)src/StochOpt.jl")
pyplot()

## Path settings
folder_path = "$(path)experiments/theory_practice_SVRG/replot/exp2a/"

## Experiments settings
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

# idx_prob = 1
for idx_prob in problems
    data = datasets[idx_prob]
    scaling = scalings[idx_prob]
    lambda = lambdas[idx_prob]

    ## Loading the data
    println("--- Loading data ---")
    data_path = "$(path)data/";
    X, y = loadDataset(data_path, data)

    ## Setting up the problem
    println("\n--- Setting up the selected problem ---")
    options = set_options(tol=10.0^(-6), max_iter=10^8, max_epocs=10^4, max_time=60.0, skip_error_calculation=10^5, batchsize=1,
                          regularizor_parameter="normalized", initial_point="zeros", force_continue=false)

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

    ## Loading data
    suffix = "lame23"
    filename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
    filename = string(filename, "-exp2a-$(suffix)-$(details)")
    OUTPUTS = load("$(folder_path)old_data/$(filename).jld", "OUTPUTS")

    ## Change legend
    # Legend Bubeck 2a
    legend1 = string(OUTPUTS[1].name)
    # str_m_bubeck = split(legend1)[4][1:end-1]
    # str_step_bubeck = split(legend1)[end][1:end-2]
    # OUTPUTS[1].name = latexstring("SVRG-Bubeck \$(m_{Bubeck}^* = $str_m_bubeck, b = 1, \\alpha_{Bubeck}^* = $str_step_bubeck)\$")
    # OUTPUTS[1].name = latexstring("SVRG \$(m^* = $str_m_bubeck, b = 1, \\alpha^* = $str_step_bubeck)\$")
    OUTPUTS[1].name = latexstring("SVRG \$(b = 1, m = 20L_{\\max}/\\mu)\$")

    # Legend Free-SVRG 2a
    legend2 = string(OUTPUTS[2].name)
    # str_m_free = split(legend2)[6][1:end-1]
    # str_step_free = split(legend2)[end][1:end-2]
    # OUTPUTS[2].name = latexstring("Free-SVRG \$(m = n = $str_m_free, b = 1, \\alpha_{Free}^*(1) = $str_step_free)\$")
    # OUTPUTS[2].name = latexstring("Free-SVRG \$(m = n = $str_m_free, b = 1, \\alpha^*(1) = $str_step_free)\$")
    OUTPUTS[2].name = latexstring("Free-SVRG \$(b = 1, m = n)\$")

    # Legend L-SVRG-D 2a
    legend3 = string(OUTPUTS[3].name)
    # str_proba_decreasing = split(legend3)[6][1:end-1]
    # str_step_decreasing = split(legend3)[end][1:end-2]
    # OUTPUTS[3].name = latexstring("L-SVRG-D \$(p = 1/n = $str_proba_decreasing, b = 1, \\alpha_{Decrease}^*(1) = $str_step_decreasing)\$")
    # OUTPUTS[3].name = latexstring("L-SVRG-D \$(p = 1/n = $str_proba_decreasing, b = 1, \\alpha^*(1) = $str_step_decreasing)\$")
    OUTPUTS[3].name = latexstring("L-SVRG-D \$(b = 1, p = 1/n)\$")

    ## Replot
    pyplot()
    # plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp2a-extract-legend", path=folder_path, legendpos=:topright, legendfont=16)
    plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp2a-submission", path=folder_path, legendpos=:topright, legendfont=10, nolegend=true)
end

