using JLD
using Plots
using StatsBase
using Match
using LaTeXStrings
include("../src/StochOpt.jl")
## Basic parameters
maxiter=10^6;
max_time = 5;
max_epocs = 20;
printiters = true;
exacterror =true;
repeat = false;       # repeat the grid_search calculation for finding the stepsize
tol = 10.0^(-8.0);
skip_error_calculation =1.0;   # number of iterations where error is not calculated (to save time!). Use 0 for default value
rep_number = 1;# number of times the optimization should be repeated. This is because of julia just in time compiling
options = MyOptions(tol,Inf,maxiter,skip_error_calculation,max_time,max_epocs,
printiters,exacterror,0,"normalized",0.0,false, false,rep_number,0)
options.embeddim = 5; # The max dimension of the S embedding matrix
## load problem
datapath = ""#
probname = "a9a";   # Data tested in paper: gisette_scale   madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders_scale
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
options.batchsize =prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
######
method_name = "BFGS";
# options.stepsize_multiplier = 100;
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
######
method_name = "SVRG";
output= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output];
######
default_path = "./data/";   savename= replace(prob.name, r"[\/]", "-");
# save("$(default_path)$(savename).jld", "OUTPUTS",OUTPUTS);

pgfplots()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS,prob,options,20) # Plot and save output
