"""
### "Towards closing the gap between the theory and practice of SVRG", O. Sebbouh, S. Jelassi, N. Gazagnadou, F. Bach, R. M. Gower (2019)

## --- EXPERIMENT 5 ---
Goal: Plotting the step size as a function of the batch size for Free-SVRG for real data sets.

## --- THINGS TO CHANGE BEFORE RUNNING ---


## --- HOW TO RUN THE CODE ---
To run this experiment, open a terminal, go into the "StochOpt.jl/" repository and run the following command:
>julia repeat_paper_experiments/repeat_theory_practice_SVRG_paper_experiment_stepsizes.jl

## --- EXAMPLE OF RUNNING TIME ---

## --- SAVED FILES ---

"""

## Bash input
machine = ARGS[1]

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
pyplot() # No problem with pyplot when called in @everywhere statement


## Create saving directories if not existing
save_path = "$(path)experiments/theory_practice_SVRG/"
#region
if !isdir(save_path)
    mkdir(save_path)
end
save_path = "$(save_path)exp5/"
if !isdir(save_path)
    mkdir(save_path)
end
if !isdir("$(save_path)figures/")
    mkdir("$(save_path)figures/")
end
#endregion


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

idx_prob = 7
data = datasets[idx_prob]
scaling = scalings[idx_prob]
lambda = lambdas[idx_prob]
# println("EXPERIMENT : ", idx_prob, " over ", length(problems))
@printf "Inputs: %s + %s + %1.1e \n" data scaling lambda

## Loading the data
println("--- Loading data ---")
data_path = "$(path)data/"
X, y = loadDataset(data_path, data)

## Setting up the problem
println("\n--- Setting up the selected problem ---")
options = set_options(tol=10.0^(-6), max_iter=10^8,
                      max_epocs=10^8,
                      max_time=60.0*60.0*24.0,
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
Lmax = prob.Lmax
L = prob.L

minibatch_seq = 1:n

free_step_size_seq = ( (n-1).*minibatch_seq ) ./ ( 6*(Lmax.*(n.-minibatch_seq) .+ n*L.*(minibatch_seq .- 1)) )

## Plot step sizes of Free-SVRG
fontsmll = 8
fontmed = 12
fontlegend = 13
fontbig = 18
xlabeltxt = "mini-batch size"

xtickpos = [1, n]
xticklabels = ["1", "n=$(n)  "]
ytickpos = [3*10^(-6), 1*10^(-3), 2*10^(-3), 3*10^(-3)]
yticklabels = ["\$3.10^{-6}\$", "\$1.10^{-3}\$", "\$2.10^{-3}\$", "\$3.10^{-3}\$"]

p = plot(minibatch_seq, free_step_size_seq,
     legend=false,
     xlabel=xlabeltxt,
     xticks=(xtickpos, xticklabels),
     yticks=(ytickpos, yticklabels),
    #  xrotation=rad2deg(pi/6),
     ylabel="step size",
     tickfont=font(fontmed),
     guidefont=font(fontbig),
     linewidth=4,
     grid=false,
     color=:blue,
     linestyle=:auto)
probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
savename = "-stepsize_vs_minibatch"
savefig(p, "$(save_path)figures/$(probname)$(savename).pdf")

println("\n\n--- EXPERIMENT 5 FINISHED ---")