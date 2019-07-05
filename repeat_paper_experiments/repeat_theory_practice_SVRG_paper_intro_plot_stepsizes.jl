"""
### "Towards closing the gap between the theory and practice of SVRG", O. Sebbouh, S. Jelassi, N. Gazagnadou, F. Bach, R. M. Gower (2019)

## --- STEP VS MINI-BATCH SIZE PLOT ---
Goal: Plotting the step size as a function of the batch size for Free-SVRG for real data sets.

## --- THINGS TO CHANGE BEFORE RUNNING ---


## --- HOW TO RUN THE CODE ---
To run this experiment, open a terminal, go into the "StochOpt.jl/" repository and run the following command:
>julia repeat_paper_experiments/repeat_theory_practice_SVRG_paper_intro_plot_stepsizes.jl

## --- EXAMPLE OF RUNNING TIME ---
Vert fast: ~ 1min

## --- SAVED FILES ---

"""

# path = "/home/nidham/phd/StochOpt.jl/"         # local
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


## Create saving directories if not existing
save_path = "$(path)experiments/theory_practice_SVRG/"
#region
if !isdir(save_path)
    mkdir(save_path)
end
#endregion

data = "slice"             # n =  53,500, d =    384
scaling = "column-scaling"
lambda = 10^(-1)
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
savefig(p, "$(save_path)$(probname)$(savename).pdf")

println("\n\n--- STEP VS MINI-BATCH SIZE PLOT FINISHED ---")