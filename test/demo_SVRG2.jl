using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")
## Basic parameters
maxiter=10^8;
max_time = 350;
max_epocs = 30;
printiters = true;
exacterror =true;
repeat = false;       # repeat the grid_search calculation for finding the stepsize
tol = 10.0^(-6.0);
skip_error_calculation =0.0;   # number of iterations where error is not calculated (to save time!). Use 0 for default value
rep_number = 2;# number of times the optimization should be repeated. This is because of julia just in time compiling
options = MyOptions(tol,Inf,maxiter,skip_error_calculation,max_time,max_epocs,
printiters,exacterror,0,"normalized",0.0,false, false,rep_number,0)
options.batchsize =100;
options.embeddim = 10; # The max number of columns of the S sketching matrix
## load problem
datapath = ""#
probname = "mushrooms";   # Data tested in paper: gisette_scale   madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders_scale
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
## Running methods
OUTPUTS = [];  # List of saved outputs
# # #
method_name = "SVRG";
output= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output];
# # # # #
method_name = "SVRG2";
output= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output];
####
method_name = "2D";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
# # #
method_name = "2Dsec";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
# ##
# # # # #
method_name = "CMgauss";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
# # #
method_name = "CMprev";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
# # # #
method_name = "DFPgauss";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
# # ##
method_name = "DFPprev";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];

default_path = "./data/";   savename= replace(prob.name, r"[\/]", "-");
save("$(default_path)$(savename).jld", "OUTPUTS",OUTPUTS);

pgfplots()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS,prob,options,20) # Plot and save output

# 
# options.batchsize = prob.numdata;
# options.skip_error_calculation =1.0;
# method_name = "grad";
# output= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
# OUTPUTS = [OUTPUTS ; output];
