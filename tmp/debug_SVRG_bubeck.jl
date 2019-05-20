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
include("./src/StochOpt.jl")

## Path settings
#region
save_path = "./experiments/Bubeck/"
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

Random.seed!(1)

## Settings to evaluate grossly the performance (error computed every 100 iterations, 50 epochs)
# options = set_options(max_iter=10^8, max_time=10000.0, max_epocs=30, initial_point="zeros", skip_error_calculation=1)

## Settings to monitor leaping (error computed at each iteration, very few epochs)
options = set_options(max_iter=20, max_time=10.0^8, max_epocs=10^8, initial_point="zeros", skip_error_calculation=1)
# options = set_options(max_iter=5, max_time=10.0^8, max_epocs=10^8, initial_point="zeros", skip_error_calculation=1)

## Load problem
datapath = "./data/"
data = "australian"     # n =     690, d = 22
# data = "ijcnn1_full"    # n = 141,691, d =
# data = "covtype_binary" # n = 581,012, d = 55
X, y = loadDataset(datapath, data)
prob = load_logistic_from_matrices(X, y, data, options, lambda=1e-3, scaling="column-scaling")

## Running methods
OUTPUTS = [] # list of saved outputs

## SVRG-Bubeck with b-nice sampling (m = m^*, b = 1, step size = gamma^*)
options.stepsize_multiplier = -1.0 # theoretical step sizes in boot_Leap_SVRG
options.batchsize = 1
sampling = build_sampling("nice", prob.numdata, options)
options.stepsize_multiplier = -1.0 # Theoretical step size in boot_SVRG_bubeck
numinneriters = 5 # 20*Lmax/mu
bubeck = initiate_SVRG_bubeck(prob, options, sampling, numinneriters=numinneriters)

# println("-------------------- WARM UP --------------------")
# # options.max_epocs = 10
# minimizeFunc(prob, bubeck, options) # Warm up
# # options.max_epocs = 100
# bubeck.reset(prob, bubeck, options)
# println("-------------------------------------------------\n")

output2 = minimizeFunc(prob, bubeck, options)
str_m_2 = @sprintf "%d" bubeck.numinneriters
str_b_2 = @sprintf "%d" bubeck.batchsize
str_step_2 = @sprintf "%.2e" bubeck.stepsize
output2.name = latexstring("$(output2.name) \$(m^* = $str_m_2, b = $str_b_2 , \\gamma^* = $str_step_2)\$")
OUTPUTS = [OUTPUTS; output2]

pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, methodname="debug_SVRG_bubeck", path=save_path, legendfont=10) # Plot and save output