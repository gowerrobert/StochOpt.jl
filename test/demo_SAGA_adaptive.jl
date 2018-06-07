using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")

## Basic parameters and options for solvers
options = set_options(max_iter=10^8, max_time = 3000.0, max_epocs = 50,  force_continue = true, initial_point = "randn"); #,repeat_stepsize_calculation =true, rep_number =10
options.batchsize =1;
options.stepsize_multiplier =1;
## load problem
datapath = ""#
probname = "w8a";   # Data tested in paper: w8a mushrooms gisette_scale,  madelon  a9a  phishing  covtype splice  rcv1_train  liver-disorders_scale
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
## Running methods
OUTPUTS = [];  # List of saved outputs

#######
sg = intiate_SAGA(prob , options, minibatch_type = "partition", probability_type= "uni")
output= minimizeFunc(prob, sg, options);
OUTPUTS = [OUTPUTS ; output];
# #####
sg = intiate_SAGA(prob , options, minibatch_type = "partition", probability_type= "opt") #
output= minimizeFunc(prob, sg, options);
OUTPUTS = [OUTPUTS ; output];
#######
options.stepsize_multiplier =1;
sg = intiate_SAGA(prob , options, minibatch_type = "partition", probability_type= "ada")
output= minimizeFunc(prob, sg, options);
OUTPUTS = [OUTPUTS ; output];


# output3= minimizeFunc_grid_stepsize(prob, "SVRG", options);
# OUTPUTS = [OUTPUTS ; output3];
# #
default_path = "./data/";   savename= replace(replace(prob.name, r"[\/]", "-"),".","_");
save("$(default_path)$(savename).jld", "OUTPUTS",OUTPUTS);
pgfplots()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS,prob,options) # Plot and save output
