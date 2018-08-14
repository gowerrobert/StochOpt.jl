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

## Running methods
OUTPUTS = [];  # List of saved outputs

####### SAGA + partition minibatch + uniform probability
options.batchsize = 100;
options.stepsize_multiplier = 10; # Why ? because it is set in the following function "initiate_SAGA"
sg = initiate_SAGA(prob, options, minibatch_type="partition", probability_type="uni");
output = minimizeFunc(prob, sg, options);
OUTPUTS = [OUTPUTS; output];

####### SAGA + nice minibatch + default ("") probability
# sg = initiate_SAGA(prob, options, minibatch_type="nice"); # what does "nice" means? : not used explicitly in initiate_SAGA
# output2 = minimizeFunc(prob, sg, options);
# OUTPUTS = [OUTPUTS; output2];

# ####### Gradrient descent
# options.batchsize = prob.numdata;
# options.skip_error_calculation = 1.0;
# output3 = minimizeFunc_grid_stepsize(prob, "grad", options);
# OUTPUTS = [OUTPUTS; output3];

####### Saving results
# default_path = "./data/";
# savename = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
# save("$(default_path)$(savename)_optimal_minibatch.jld", "OUTPUTS", OUTPUTS);

gr() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options) # Plot and save output
