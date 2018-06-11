using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")

## Basic parameters
options = set_options(max_iter=10^8, max_time = 350.0, max_epocs = 30, repeat_stepsize_calculation = true, rep_number =5);
options.batchsize =100;
options.embeddim = 10; # The max number of columns of the S sketching matrix
## load problem
datapath = ""#
probname = "mushrooms";   # Data tested in paper: gisette_scale   madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders_scale
prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
## Running methods
OUTPUTS = [];  # List of saved outputs
# # #
method_names = ["AMprev", "SVRG", "2Dsec", "AMgauss" ]  # Curvature matching methods: CMgauss,  CMprev
for method_name in method_names
    output= minimizeFunc_grid_stepsize(prob, method_name, options);
    OUTPUTS = [OUTPUTS ; output];
end
### Gradient with fixed step
options.batchsize =prob.numdata;
options.skip_error_calculation =1.0;
method_name = "grad";
output= minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS ; output];

default_path = "./data/";   savename= replace(prob.name, r"[\/]", "-");
save("$(default_path)$(savename).jld", "OUTPUTS",OUTPUTS);

pgfplots()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS,prob,options,20) # Plot and save output
