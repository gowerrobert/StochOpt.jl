using Plots
using JLD
using StatsBase
using Match
include("../src/StochOpt.jl")

## Basic parameters and options for solvers
options = set_options(max_iter=10^8, max_time=1000.0, max_epocs=150, force_continue=true, initial_point="randn"); #,repeat_stepsize_calculation =true, rep_number =10

## Load problem
datapath = ""
probname = "splice";   # Data tested in paper: w8a mushrooms gisette_scale,  madelon  a9a  phishing  covtype splice  rcv1_train  liver-disorders_scale
prob = load_logistic(probname, datapath, options);  # Loads logisitc problem
# options.stepsize_multiplier = 10; # Why ? because it is set in the following function "initiate_SAGA"

## Running methods
OUTPUTS = [];  # List of saved outputs

###### SAGA + nice minibatch + default ("") probability
## Minibatch size = 1 : SGD
options.batchsize = 1;
sg = initiate_SAGA(prob, options, minibatch_type="nice");
output = minimizeFunc(prob, sg, options);
OUTPUTS = [OUTPUTS; output];

## Minibatch size = 10
options.batchsize = 10;
sg = initiate_SAGA(prob, options, minibatch_type="nice");
output = minimizeFunc(prob, sg, options);
OUTPUTS = [OUTPUTS; output];

## Minibatch size = 20
options.batchsize = 20;
sg = initiate_SAGA(prob, options, minibatch_type="nice");
output = minimizeFunc(prob, sg, options);
OUTPUTS = [OUTPUTS; output];

## Minibatch size = numdata : Gradrient descent
# options.batchsize = prob.numdata;
# options.skip_error_calculation = 1.0; # What is skip_error_calculation ??
# output3 = minimizeFunc_grid_stepsize(prob, "grad", options);
# OUTPUTS = [OUTPUTS; output3];

## Saving results into Julia Data format
# default_path = "./data/";
# savename = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
# save("$(default_path)$(savename)_optimal_minibatch.jld", "OUTPUTS", OUTPUTS);

## Saving plots
prob.name = string(prob.name, "_optimal_minibatch");
gr() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options) # Plot and save output