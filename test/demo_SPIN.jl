using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")


#region
save_path = "./experiments/SPIN/"
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

## This is a basic demo showing how to setup and call different optimization methods for the ERM problem
## Basic parameters and options for solvers
options = set_options(max_iter=10^8, max_time=300.0, max_epocs=500, force_continue=true, initial_point="zeros",exacterror = true); #repeat_stepsize_calculation =true, rep_number =10
## load problem
datapath = "./data/"
data = "mushrooms"; # Data tested in paper: w8a mushrooms gisette_scale,  madelon  a9a  phishing  covtype splice  rcv1_train  liver-disorders_scale
X, y = loadDataset(datapath, data)
#  X, y, probname = gen_gauss_data(numfeatures, numdata);
# Scaling the regularizor

lambda = 1.0;
prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling = "intersept"); #scaling="none" "column-scaling"

sigmamax =    (prob.L - lambda)*4*length(y);
Lhat = (0.25*sigmamax)/(lambda*prob.numdata)  +1;
options.stepsize_multiplier = 1/(5*Lhat);


# sketchsize= 10;
# C = convert(Array{Int64}, sample(1:prob.numfeatures, sketchsize, replace=false));
# gC = zeros(sketchsize);
# SHS = zeros(sketchsize,sketchsize);
# x = zeros(prob.numfeatures);
# prob.Hess_CC_g_C!(x, 1:prob.numdata, C, gC, SHS);


options.batchsize = prob.numdata;
options.skip_error_calculation = 1;
## Running methods
OUTPUTS = [];  # List of saved outputs
#######

## Most methods are called by name, for example the SVRG
smallsketch =convert(Int64,ceil(log(prob.numfeatures)))
bigsketch = convert(Int64,ceil(sqrt(prob.numfeatures)))
#
## Coordinate sketchs
spin = initiate_SPIN(prob, options, sketchsize=bigsketch, sketchtype="coord" )
output = minimizeFunc(prob, spin, options);
OUTPUTS = [OUTPUTS; output];
#
spin = initiate_SPIN(prob, options, sketchsize=bigsketch)
output = minimizeFunc(prob, spin, options);
OUTPUTS = [OUTPUTS; output];

spin = initiate_SPIN(prob, options, sketchsize=bigsketch, sketchtype="prev" )
output = minimizeFunc(prob, spin, options);
OUTPUTS = [OUTPUTS; output];

# # There are also several full batch methods such as gradient descent
# method_name = "BFGS";
# output1= minimizeFunc_grid_stepsize(prob, method_name, options);
# OUTPUTS = [OUTPUTS ; output1];

# #plot and save graphs
# gr()# gr() pyplot() # pgfplots() #plotly()
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
savename = string(savename, "-", "demo_SPIN")
save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

# pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, methodname="demo_SPIN", path=save_path) # Plot and save output


# spin = initiate_SPIN(prob, options, sketchsize=5, sketchtype="cov", weight= 1.0)  
# options.stepsize_multiplier = 0.009;
# output = minimizeFunc(prob, spin, options);
# OUTPUTS = [OUTPUTS; output];


# spin = initiate_SPIN(prob, options, sketchsize=smallsketch, sketchtype="prev", weight= 0.05)
# options.stepsize_multiplier = 0.009;
# output = minimizeFunc(prob, spin, options);
# OUTPUTS = [OUTPUTS; output];

# spin = initiate_SPIN(prob, options, sketchsize=smallsketch, sketchtype="prev", weight= 0.05)
# output = minimizeFunc(prob, spin, options);
# OUTPUTS = [OUTPUTS; output];
# #
# spin = initiate_SPIN(prob, options, sketchsize=smallsketch, sketchtype="prev", weight= 0.01)
# output = minimizeFunc(prob, spin, options);
# OUTPUTS = [OUTPUTS; output];
# #
# spin = initiate_SPIN(prob, options, sketchsize=bigsketch, sketchtype="prev", weight= 0.05)
# output = minimizeFunc(prob, spin, options);
# OUTPUTS = [OUTPUTS; output];
#
# spin = initiate_SPIN(prob, options, sketchsize=bigsketch, sketchtype="prev", weight= 0.01)
# output = minimizeFunc(prob, spin, options);
# OUTPUTS = [OUTPUTS; output];
# #

# # output= minimizeFunc_grid_stepsize(prob, spin, options);