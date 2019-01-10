using Plots
using JLD
using StatsBase
using Match
include("../src/StochOpt.jl")

## Basic parameters and options for solvers
options = set_options(max_iter=10^8, max_time=1000.0, max_epocs=150, force_continue=true,  repeat_stepsize_calculation=false, initial_point="randn"); #,repeat_stepsize_calculation =true, rep_number =10
options.batchsize = 100;
## load problem
datapath = "./data/";
probname = "mushrooms";   # Data tested in paper: w8a mushrooms gisette_scale,  madelon  a9a  phishing  covtype splice  rcv1_train  liver-disorders_scale
prob = load_logistic(datapath, probname, options);  # Loads logisitc problem
## Running methods
OUTPUTS = [];  # List of saved outputs
#######
options.stepsize_multiplier = 10;
sg = initiate_SAGA_partition(prob, options, minibatch_type="partition", probability_type="uni")
output = minimizeFunc_grid_stepsize(prob, sg, options);
OUTPUTS = [OUTPUTS; output];
# #
# sg.unbiased = false;
sg = initiate_SAGA_partition(prob, options, minibatch_type="partition", probability_type="opt")
output = minimizeFunc_grid_stepsize(prob, sg, options);
OUTPUTS = [OUTPUTS; output];
######
SAGA_nice = initiate_SAGA_nice(prob, options); # separated implementation from SAGA
output = minimizeFunc_grid_stepsize(prob, SAGA_nice, options);
OUTPUTS = [OUTPUTS; output];
#######
output3 = minimizeFunc_grid_stepsize(prob, "SVRG", options);
OUTPUTS = [OUTPUTS; output3];
#
options.batchsize = prob.numdata;
options.skip_error_calculation = 1.0
output3 = minimizeFunc_grid_stepsize(prob, "grad", options);
OUTPUTS = [OUTPUTS; output3];
# #
default_path = "./data/";
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
save("$(default_path)$(savename).jld", "OUTPUTS", OUTPUTS);

gr()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options) # Plot and save output
