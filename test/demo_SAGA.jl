using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")

## Basic parameters and options for solvers
options = set_options(max_iter=10^8, max_time = 500.0, max_epocs = 30,  force_continue = true); #,repeat_stepsize_calculation =true, rep_number =10
options.batchsize =100;
## load problem
datapath = ""#
probname = "madelon";   # Data tested in paper: w8a mushrooms gisette_scale,  madelon  a9a  phishing  covtype splice  rcv1_train  liver-disorders_scale
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
## Running methods
OUTPUTS = [];  # List of saved outputs
# #####
SAGA = intiate_SAGA(prob , options, minibatch_type = "partition", probability_type= "opt")
options.stepsize_multiplier =8;
output= minimizeFunc(prob, SAGA, options);
OUTPUTS = [OUTPUTS ; output];
#
SAGA.unbiased =false;
SAGA.name = "SAG-100-opt";
options.stepsize_multiplier =8;
output= minimizeFunc(prob, SAGA, options);
OUTPUTS = [OUTPUTS ; output];
# ######
# SAGA = intiate_SAGA(prob , options, minibatch_type = "nice")
# output= minimizeFunc(prob, SAGA, options);
# OUTPUTS = [OUTPUTS ; output];
# #######
# # # # # #
options.embeddim =10;
output3= minimizeFunc_grid_stepsize(prob, "DFPprev", options);
OUTPUTS = [OUTPUTS ; output3];

output3= minimizeFunc_grid_stepsize(prob, "SVRG", options);
OUTPUTS = [OUTPUTS ; output3];

options.batchsize =prob.numdata;
options.skip_error_calculation =1.0
output3= minimizeFunc_grid_stepsize(prob, "grad", options);
OUTPUTS = [OUTPUTS ; output3];
#
default_path = "./data/";   savename= replace(replace(prob.name, r"[\/]", "-"),".","_");
save("$(default_path)$(savename).jld", "OUTPUTS",OUTPUTS);

pyplot()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS,prob,options) # Plot and save output
