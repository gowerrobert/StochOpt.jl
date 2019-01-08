using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")

## Basic parameters
options = set_options(max_iter=10^8, max_time=350.0, max_epocs=30, repeat_stepsize_calculation=true, rep_number=1);
options.batchsize = 100;
options.embeddim = 10; # The max number of columns of the S sketching matrix
## load problem
datapath = "./data/"
probname = "mushrooms"; # Data tested in paper: gisette_scale   madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders_scale
prob = load_logistic(datapath, probname, options);  # Loads logisitc problem
## Running methods
# # #
method_names = ["AMprev", "SVRG", "2Dsec", "AMgauss", "SVRG2"]  # Curvature matching methods: CMgauss,  CMprev
global OUTPUTS =[];  # List of saved outputs
for method_name in method_names
    output = minimizeFunc_grid_stepsize(prob, method_name, options);
    OUTPUTS = [OUTPUTS; output];
end
### Gradient with fixed step
options.batchsize = prob.numdata;
options.skip_error_calculation = 1.0;
method_name = "grad";
output = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output];

default_path = "./data/";
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
save("$(default_path)$(savename).jld", "OUTPUTS", OUTPUTS);

gr()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, 20) # Plot and save output
