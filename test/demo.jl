using Plots
using JLD, StatsBase, Match
include("../src/StochOpt.jl")
## This is a basic demo showing how to setup and call different optimization methods for the ERM problem
## Basic parameters and options for solvers
options = set_options(max_iter=10^8, max_time=2.0, max_epocs=50, force_continue=true, initial_point="randn"); #repeat_stepsize_calculation =true, rep_number =10
options.batchsize = 100;
## load problem
data_path = "./data/"
probname = "mushrooms"; # Data tested in paper: w8a mushrooms gisette_scale,  madelon  a9a  phishing  covtype splice  rcv1_train  liver-disorders_scale
## Loads logisitc problem
prob = load_logistic(data_path, probname, options);

## Running methods
OUTPUTS = [];  # List of saved outputs
#######

## Most methods are called by name, for example the SVRG
output = minimizeFunc_grid_stepsize(prob, "SVRG", options);
OUTPUTS = [OUTPUTS; output];

## or the "Action Matching" method call AMprev from [1]
options.embeddim = 10;
output = minimizeFunc_grid_stepsize(prob, "AMprev", options);
OUTPUTS = [OUTPUTS; output];

# The SAGA methods need to be initited before by calling "initiate_SAGA" for example
sg = initiate_SAGA(prob , options, minibatch_type="partition", probability_type="opt") # Saga with optimal sampling probabilities as given in [4]
output = minimizeFunc_grid_stepsize(prob, sg, options); # then pass the type sg
OUTPUTS = [OUTPUTS; output];
#######

# If you don't want to use a grid search to determine the stepsize, you can call "minimizeFunc" directly and specify a stepsize_multiplier (it multiplies a baseline stepsize such as 1/L)
options.stepsize_multiplier = 25;
sg = initiate_SAGA(prob, options, minibatch_type="partition", probability_type="uni")  # Saga with uniform sampling
output = minimizeFunc(prob, sg, options); # then pass the type sg
OUTPUTS = [OUTPUTS; output];
#######


# There are also several full batch methods such as gradient descent
options.batchsize = prob.numdata; # Use full batch
method_name = "BFGS";
output1 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output1];

# and even an accelerated BFGS method
options.embeddim = [prob.numdata, 1/prob.numfeatures]; # Acceleration parameters
method_name = "BFGS_accel";
output3 = minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS; output3];

# saving the data for later
default_path = "./data/";
savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
save("$(default_path)$(savename).jld", "OUTPUTS", OUTPUTS);

#plot and save graphs
pgfplots()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options) # Plot and save output

# References
#
# [1] Tracking the gradients using the Hessian: A new look at variance reducing stochastic methods
# RMG, Nicolas Le Roux and Francis Bach. To appear in AISTATS 2018
#
# [2] Accelerated stochastic matrix inversion: general theory and speeding up BFGS rules for faster second-order optimization
# RMG, Filip Hanzely, P. Richtárik and S. Stich. arXiv:1801.05490, 2018
#
# [3] LIBSVM : a library for support vector machines.
# Chih-Chung Chang and Chih-Jen Lin, ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011. Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm
#
# [4] Stochastic Quasi-Gradient Methods: Variance Reduction via Jacobian Sketching
# RMG, Peter Richtárik, Francis Bach
