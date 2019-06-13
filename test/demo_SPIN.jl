using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")


#region
save_path = "./experiments/SPIN/"
if !isdir(save_path) # create directory if not existing
    if !isdir("./experiments/")
        mkdir("./experiments/")
    end
    mkdir(save_path)
    mkdir("$(save_path)data/")
    mkdir("$(save_path)figures/")
end

if !isdir("$(save_path)data/")
    mkdir("$(save_path)data/")
end

if !isdir("$(save_path)figures/")
    mkdir("$(save_path)figures/")
end
#endregion

## This is a basic demo showing how to setup and call different optimization methods for the ERM problem
## Basic parameters and options for solvers
options = set_options(max_iter=10^8, max_time=300.0, max_epocs=400, force_continue=true, initial_point="randn"); #repeat_stepsize_calculation =true, rep_number =10
## load problem
datapath = "./data/"
data = "australian"; # Data tested in paper: w8a mushrooms gisette_scale,  madelon  a9a  phishing  covtype splice  rcv1_train  liver-disorders_scale
X, y = loadDataset(datapath, data)

# Scaling the regularizor
Lmax = maximum(sum(prob.X.^2,1));
lambda = Lmax/n;

prob = load_logistic_from_matrices(X, y, data, options, lambda=lamda, scaling="column-scaling")
## Loads logisitc problem
# numdata = 20;
# numfeatures =4;
#  X, y, probname = gen_gauss_data(numfeatures, numdata);
# prob =  load_logistic_from_matrices(X, y, probname, options, lambda=1e-1, scaling="column-scaling")#load_ridge_regression(X,y,probname,options);
# d, n = size(X);



options.batchsize = prob.numdata;
options.skip_error_calculation = 10;
## Running methods
OUTPUTS = [];  # List of saved outputs
#######


# spin = initiate_SPIN(prob, options, sketchsize=5, sketchtype="cov", weight= 1.0)  
# options.stepsize_multiplier = 0.009;
# output = minimizeFunc(prob, spin, options);
# # output= minimizeFunc_grid_stepsize(prob, spin, options);
# OUTPUTS = [OUTPUTS; output];
## Most methods are called by name, for example the SVRG
smallsketch =convert(Int64,ceil(log(prob.numfeatures)))
bigsketch = convert(Int64,ceil(sqrt(prob.numfeatures)))
options.stepsize_multiplier = 0.01;
# spin = initiate_SPIN(prob, options, sketchsize=smallsketch, sketchtype="prev", weight= 0.05)
# options.stepsize_multiplier = 0.009;
# output = minimizeFunc(prob, spin, options);
# # output= minimizeFunc_grid_stepsize(prob, spin, options);
# OUTPUTS = [OUTPUTS; output];

spin = initiate_SPIN(prob, options, sketchsize=smallsketch, sketchtype="prev", weight= 0.05)
output = minimizeFunc(prob, spin, options);
# output= minimizeFunc_grid_stepsize(prob, spin, options);
OUTPUTS = [OUTPUTS; output];
#
spin = initiate_SPIN(prob, options, sketchsize=smallsketch, sketchtype="prev", weight= 0.01)
output = minimizeFunc(prob, spin, options);
# output= minimizeFunc_grid_stepsize(prob, spin, options);
OUTPUTS = [OUTPUTS; output];
#
spin = initiate_SPIN(prob, options, sketchsize=bigsketch, sketchtype="prev", weight= 0.05)
output = minimizeFunc(prob, spin, options);
# output= minimizeFunc_grid_stepsize(prob, spin, options);
OUTPUTS = [OUTPUTS; output];
#
spin = initiate_SPIN(prob, options, sketchsize=bigsketch, sketchtype="prev", weight= 0.01)
output = minimizeFunc(prob, spin, options);
# output= minimizeFunc_grid_stepsize(prob, spin, options);
OUTPUTS = [OUTPUTS; output];
#
spin = initiate_SPIN(prob, options, sketchsize=smallsketch, sketchtype="prev", weight= 0.0)
output = minimizeFunc(prob, spin, options);
# output= minimizeFunc_grid_stepsize(prob, spin, options);
OUTPUTS = [OUTPUTS; output];
#
spin = initiate_SPIN(prob, options, sketchsize=smallsketch )
output = minimizeFunc(prob, spin, options);
# output = minimizeFunc_grid_stepsize(prob, spin, options);
OUTPUTS = [OUTPUTS; output];

# spin = initiate_SPIN(prob, options, sketchsize=prob.numfeatures)
# options.stepsize_multiplier = 0.04;
# output = minimizeFunc(prob, spin, options);
# # output = minimizeFunc_grid_stepsize(prob, spin, options);
# OUTPUTS = [OUTPUTS; output];


# There are also several full batch methods such as gradient descent
# method_name = "BFGS";
# output1= minimizeFunc_grid_stepsize(prob, method_name, options);
# OUTPUTS = [OUTPUTS ; output1];


# saving the data for later
# default_path = "./data/";
# savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
# save("$(default_path)$(savename).jld", "OUTPUTS", OUTPUTS);

# #plot and save graphs
# gr()# gr() pyplot() # pgfplots() #plotly()
# plot_outputs_Plots(OUTPUTS, prob, options) # Plot and save output
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")
savename = string(savename, "-", "demo_SPIN")
save("$(save_path)data/$(savename).jld", "OUTPUTS", OUTPUTS)

pyplot() # gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, methodname="demo_SPIN", path=save_path) # Plot and save output
