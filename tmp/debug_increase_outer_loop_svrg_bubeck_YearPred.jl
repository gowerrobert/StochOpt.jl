# Debug SVRG-Bubeck for YearPrediction lmbd = 0.1

## General settings
max_epochs = 2 # at 16 occurs the divergence
max_time = 60.0*60.0*24.0
precision = 10.0^(-6)

## File names
details = "debug"

path = "/home/nidham/phd/StochOpt.jl/"
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
pyplot()

## Path settings
save_path = "$(path)experiments/theory_practice_SVRG/exp1a/"

datasets = ["YearPredictionMSD_full"]
scalings = ["column-scaling"]
lambdas = [10^(-1)]
skip_errors = [[50000 30000 -2. 20000]]

# 3)  YearPredictionMSD_full + scaled + 1e-1  25/06 11:14 / 16 epochs => WARNING: potential bug spotted:
# From worker 4:	   3450000  |           0.01132548964563050847           |    14.39  |  47049.4448  |
# From worker 4:	   3500000  |           0.01224447809451410275           |    14.58  |  47630.2359  |
# From worker 4:	   3550000  |           0.00970780021823384846           |    14.78  |  48605.9720  |
# From worker 4:	   3600000  |           0.01258959501736736653           |    14.97  |  49200.2751  |
# From worker 4:	SVRG-Bubeck outer loop at iteration: 3644286
# From worker 4:	   3644286  |           0.52289196287182837519           |    16.14  |  50079.2315  |
# ====> Divergence at last point, during the outer loop

data = datasets[1]
scaling = scalings[1]
lambda = lambdas[1]
skip_error = skip_errors[1]
@printf "Inputs: %s + %s + %1.1e \n" data scaling lambda

#region
Random.seed!(1)

## Loading the data
println("--- Loading data ---")
data_path = "$(path)data/";
X, y = loadDataset(data_path, data)

## Setting up the problem
println("\n--- Setting up the selected problem ---")
options = set_options(tol=precision, max_iter=10^8,
                      max_epocs=max_epochs,
                      max_time=max_time,
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

X = nothing
y = nothing
n = prob.numdata
#endregion

## Running methods
OUTPUTS = [] # list of saved outputs

################################################################################
################################# SVRG-BUBECK ##################################
################################################################################
## SVRG-Bubeck with 1-nice sampling ( m = m^*, b = 1, step size = gamma^* )
numinneriters = -1                             # theoretical inner loop size (m^* = 20*Lmax/mu) set in initiate_SVRG_bubeck
options.batchsize = 1                          # mini-batch size set to 1
options.stepsize_multiplier = -1.0             # theoretical step size (gamma^* = 1/10*Lmax) set in boot_SVRG_bubeck
sampling = build_sampling("nice", n, options)
bubeck = initiate_SVRG_bubeck(prob, options, sampling, numinneriters=numinneriters)

## Setting the number of skipped iteration
options.skip_error_calculation = skip_error[1] # skip error different for each algo

println("-------------------- WARM UP --------------------")
tmp = options.max_epocs
options.max_epocs = 1
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
options.max_epocs = max_epochs
println("\n")

## Saving outputs and plots
if path == "/home/infres/ngazagnadou/StochOpt.jl/"
    suffix = "lame23"
else
    suffix = ""
end
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
savename = string(savename, "-exp1a-$(suffix)-$(details)")
save("$(save_path)outputs/$(savename).jld", "OUTPUTS", OUTPUTS)

pyplot()
# plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp1a-$(suffix)-$(max_epochs)_max_epochs", path=save_path, legendpos=:topright, legendfont=6) # Plot and save output
plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp1a-$(suffix)-$(details)", path=save_path, nolegend=true)

println("\nSTRONG CONVEXITY : ", prob.mu, "\n")

end
println("\n\n--- EXPERIMENT 1.A FINISHED ---")