using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")
## This is a basic demo showing how to setup and call different optimization methods for the ERM problem
## Basic parameters and options for solvers
options = set_options(max_iter=10^8, max_time=3.0, max_epocs=2000, force_continue=true, initial_point="randn"); #repeat_stepsize_calculation =true, rep_number =10
## load problem
datapath = "./data/"
probname = "splice"; # Data tested in paper: w8a mushrooms gisette_scale,  madelon  a9a  phishing  covtype splice  rcv1_train  liver-disorders_scale
## Loads logisitc problem
# numdata = 1000;
# numfeatures =10;
#  X, y, probname = gen_gauss_data(numfeatures, numdata);
# prob =  load_ridge_regression(X,y,probname,options);
prob = load_logistic(datapath, probname, datapath, options);
options.batchsize = prob.numdata;
options.skip_error_calculation = 10;
## Running methods
OUTPUTS = [];  # List of saved outputs
#######

## Most methods are called by name, for example the SVRG
spin = initiate_SPIN(prob, options, sketchsize=5)
options.stepsize_multiplier = 0.009;
output = minimizeFunc(prob, spin, options);
# output= minimizeFunc_grid_stepsize(prob, spin, options);
OUTPUTS = [OUTPUTS; output];
#
spin = initiate_SPIN(prob, options, sketchsize=5, sketchtype="prev")
options.stepsize_multiplier = 0.009;
output = minimizeFunc(prob, spin, options);
# output = minimizeFunc_grid_stepsize(prob, spin, options);
OUTPUTS = [OUTPUTS; output];

spin = initiate_SPIN(prob, options, sketchsize=prob.numfeatures)
options.stepsize_multiplier = 0.04;
output = minimizeFunc(prob, spin, options);
# output = minimizeFunc_grid_stepsize(prob, spin, options);
OUTPUTS = [OUTPUTS; output];


# There are also several full batch methods such as gradient descent
# method_name = "BFGS";
# output1= minimizeFunc_grid_stepsize(prob, method_name, options);
# OUTPUTS = [OUTPUTS ; output1];


# saving the data for later
default_path = "./data/"; savename = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
save("$(default_path)$(savename).jld", "OUTPUTS", OUTPUTS);

#plot and save graphs
gr()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options) # Plot and save output
