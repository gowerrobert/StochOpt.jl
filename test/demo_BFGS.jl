using JLD
using Plots
using StatsBase
using Match
using LaTeXStrings
include("../src/StochOpt.jl")

## Basic parameters
options = set_options(max_iter=10^6, skip_error_calculation =5, max_time = 500.0,   max_epocs = 250, repeat_stepsize_calculation = false, rep_number =2);
## load problem
datapath = ""#
probname = "mushrooms";   # Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
options.batchsize =prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1= minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS ; output1];
#####
options.embeddim = [prob.numdata, 1/prob.numfeatures];#  = [mu, nu],
method_name = "BFGS_accel";
output1= minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS ; output1];
#####
method_name = "grad";
output3= minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS ; output3];

pgfplots() # pgfplots
plot_outputs_Plots(OUTPUTS,prob,options,options.max_epocs) # Plot and save output # max_epocs
