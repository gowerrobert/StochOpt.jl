using JLD
using Plots
using StatsBase
using Match
using LaTeXStrings
include("../src/StochOpt.jl")

## Basic parameters
options = set_options(tol=10.0^(-16.0), skip_error_calculation=5, max_iter=10^6, max_time=500.0,
        max_epocs=200, repeat_stepsize_calculation=false, rep_number=2);

## load problem
datapath = ""
gr() # gr()
# Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders

#########################################################################################
######################### We should do a loop over the problems #########################
#########################################################################################

##################
## splice problem
##################
probname = "splice";
prob = load_logistic(probname, datapath, options);  # Loads logisitc problem
options.batchsize = prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1= minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output1];#
###
options.embeddim =  [prob.numdata, 0.9];
method_name = "BFGS_accel";
output3 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS, prob, options, options.max_epocs)

##################
## australian problem
##################
probname = "australian";
prob = load_logistic(probname, datapath, options);  # Loads logisitc problem
options.batchsize = prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output1];
options.embeddim = [30, 0.5];
###
method_name = "BFGS_accel";
output3 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS, prob, options, options.max_epocs)

##################
##  phishing problem
##################
probname = "phishing";
prob = load_logistic(probname, datapath, options);  # Loads logisitc problem
options.batchsize = prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output1];
###
options.embeddim = [prob.numdata/9, 7/prob.numfeatures];
method_name = "BFGS_accel";
output3 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS, prob, options, options.max_epocs)

##################
##  mushrooms problem
##################
probname = "mushrooms";
prob = load_logistic(probname, datapath, options);  # Loads logisitc problem
options.batchsize = prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output1];
###
options.embeddim = [prob.numdata, 1/prob.numfeatures];
method_name = "BFGS_accel";
output3 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS, prob, options, options.max_epocs)

##################
## Madelon problem
##################
probname = "madelon";
prob = load_logistic(probname, datapath, options);  # Loads logisitc problem
options.batchsize = prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output1];
options.embeddim = [800,0.1];   # = [mu, nu]
###
method_name = "BFGS_accel";
output3 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS, prob, options, options.max_epocs) # Plot and save output # max_epocs

##################
## a9a problem
##################
probname = "a9a";
prob = load_logistic(probname, datapath, options);  # Loads logisitc problem
options.batchsize = prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output1];
###
method_name = "BFGS_accel";
options.embeddim = [prob.numdata, 0.01];  # = [mu, nu]
output3 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS, prob, options, options.max_epocs) # Plot and save output # max_epocs

#################
# w8a problem
##################
probname = "w8a";   # Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders
prob =  load_logistic(probname, datapath, options);  # Loads logisitc problem
options.batchsize = prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output1];
# ###
options.embeddim = [2*prob.numdata, 0.05]; # = [mu, nu]
method_name = "BFGS_accel";
output3 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output3];
# ### plotting and saving
plot_outputs_Plots(OUTPUTS, prob, options, options.max_epocs) # Plot and save output # max_epocs

##################
## covtype problem
##################
probname = "covtype";   # Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders
prob = load_logistic(probname, datapath, options);  # Loads logisitc problem
options.batchsize = prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output1];
###
options.embeddim = [prob.numdata, 1/prob.numfeatures]; # = [mu, nu]
method_name = "BFGS_accel";
output3 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS, prob, options, options.max_epocs)