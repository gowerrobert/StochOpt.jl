using JLD
using Plots
using StatsBase
using Match
using LaTeXStrings
include("../src/StochOpt.jl")

## Basic parameters
options = set_options(max_iter=10^6, skip_error_calculation=5, max_time=500.0, max_epocs=250, repeat_stepsize_calculation=false, rep_number=2);
## load problem
datapath = "./data/";
probname = "mushrooms";   # Data tested in paper: w8a mushrooms gisette_scale,  madelon  a9a  phishing  covtype splice  rcv1_train  liver-disorders_scale
X, y = loadDataset(datapath,probname);
prob = load_logistic_from_matrices(X, y, probname, options, lambda=1e-1, scaling="none");
options.batchsize = prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output1];
#####
options.embeddim = [prob.numdata, 1/prob.numfeatures];#  = [mu, nu],
method_name = "BFGS_accel";
output2 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output2];
#####
method_name = "grad";
output3 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output3];

gr() # pgfplots
plot_outputs_Plots(OUTPUTS, prob, options, datapassbnd = options.max_epocs) # Plot and save output # max_epocs
# OUTPUTS, prob::Prob, options ; datapassbnd::Int64=0, suffix::AbstractString=""
