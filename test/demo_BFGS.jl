using JLD
using Plots
using StatsBase
using Match
using LaTeXStrings
include("../src/StochOpt.jl")
## Basic parameters
maxiter=10^6;
max_time = 500;
max_epocs = 200;
printiters = true;
exacterror =true;
repeat = false;       # repeat the grid_search calculation for finding the stepsize
tol = 10.0^(-16.0);
skip_error_calculation =5.0;   # number of iterations where error is not calculated (to save time!). Use 0 for default value
rep_number = 1;# number of times the optimization should be repeated. This is because of julia just in time compiling
options = MyOptions(tol,Inf,maxiter,skip_error_calculation,max_time,max_epocs,
printiters,exacterror,0,"normalized",0.0,false, false,rep_number,0)
## load problem
datapath = ""#
probname = "mushrooms";   # Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
options.batchsize =prob.numdata;  # full batch
## Running methods
OUTPUTS = [];  # List of saved outputs
method_name = "BFGS";
output1= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output1];
options.embeddim =  [1.8, 4];
###
method_name = "BFGS_accel";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
######
method_name = "grad";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];

pgfplots()
plot_outputs_Plots(OUTPUTS,prob,options,max_epocs) # Plot and save output # max_epocs



## Junk code for generating theoretical nu and mu parameters
# H0 = prob.Hess_eval(zeros(prob.numfeatures), 1:prob.numdata);
# TrH0 = trace(H0);
# nu_theo = TrH0/minimum(diag(H0));
# mu_theo = min(eigmin(H0))/TrH0;
# options.embeddim = [mu_theo, nu_theo]; # =[mu, nu]  # [prob.lambda, prob.numdata]   # [mu_theo, nu_theo];
# options.embeddim =[prob.lambda, prob.numdata];
# nu = 10^10;
# mu = 1/nu;
# options.embeddim =[mu, nu] # Sanity check, should be the same as BFGS
