using JLD
using Plots
using StatsBase
using Match
using LaTeXStrings
include("../src/StochOpt.jl")
## Basic parameters
maxiter=10^6;
max_time = 500;
max_epocs = 150;
printiters = true;
exacterror =true;
repeat = false;       # repeat the grid_search calculation for finding the stepsize
tol = 10.0^(-16.0);
skip_error_calculation =5.0;   # number of iterations where error is not calculated (to save time!). Use 0 for default value
rep_number = 2;# number of times the optimization should be repeated. This is because of julia just in time compiling
options = MyOptions(tol,Inf,maxiter,skip_error_calculation,max_time,max_epocs,
printiters,exacterror,0,"normalized",0.0,false, false,rep_number,0)
## load problem
datapath = ""#
gr()
##################
## splice problem
##################
probname = "splice";   # Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
options.batchsize =prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output1];
options.embeddim =  [30, 0.5];
###
method_name = "BFGS_accel";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS,prob,options,max_epocs)
##################
## australian problem
##################
probname = "australian";   # Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
options.batchsize =prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output1];
options.embeddim =  [30,0.5];
###
method_name = "BFGS_accel";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS,prob,options,max_epocs)
##################
##  phishing problem
##################
probname = "phishing";   # Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
options.batchsize =prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output1];
options.embeddim =  [1.05, 1.8];
###
method_name = "BFGS_accel";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS,prob,options,max_epocs)
##################
##  mushrooms problem
##################
probname = "mushrooms";   # Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
options.batchsize =prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output1];
options.embeddim = [prob.numdata, 1/prob.numfeatures];
###
method_name = "BFGS_accel";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS,prob,options,max_epocs)
##################
## Madelon problem
##################
probname = "madelon";   # Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
options.batchsize =prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output1];
options.embeddim =  [800,0.1];   # = [mu, nu]
###
method_name = "BFGS_accel";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS,prob,options,max_epocs) # Plot and save output # max_epocs
##################
## a9a problem
##################
probname = "a9a";   # Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
options.batchsize =prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output1];
###
method_name = "BFGS_accel";
options.embeddim =  [prob.numdata ,0.01];  # = [mu, nu]
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS,prob,options,max_epocs) # Plot and save output # max_epocs
##################
## w8a problem
##################
probname = "w8a";   # Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
options.batchsize =prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output1];
###
options.embeddim = [prob.numdata/2, 2/prob.numfeatures]; # = [mu, nu]
method_name = "BFGS_accel";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS,prob,options,max_epocs) # Plot and save output # max_epocs
##################
## covtype problem
##################
probname = "covtype";   # Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
options.batchsize =prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output1];
###
options.embeddim = [prob.numdata, 1/prob.numfeatures]; # = [mu, nu]
method_name = "BFGS_accel";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
### plotting and saving
plot_outputs_Plots(OUTPUTS,prob,options,max_epocs)
