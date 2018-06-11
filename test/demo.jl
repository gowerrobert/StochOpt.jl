using Plots
using JLD
using StatsBase
using Match
include("../src/StochOpt.jl")
## This is a basic demo showing how to setup and call different optimization methods for the ERM problem
## Basic parameters and options for solvers
options = set_options(max_iter=10^8, max_time = 1000.0, max_epocs = 150,  force_continue = true, initial_point ="randn"); #repeat_stepsize_calculation =true, rep_number =10
options.batchsize =100;
## load problem
datapath = ""#
probname = "mushrooms";   # Data tested in paper: w8a mushrooms gisette_scale,  madelon  a9a  phishing  covtype splice  rcv1_train  liver-disorders_scale

 # Loads logisitc problem
prob =  load_logistic(probname,datapath,options);

# Load a quadratic problem
# X, y, probname = gen_gauss_data(numfeatures, numdata, lambda = lambda_input);
# prob =   load_ridge_regression(X, y, probname, options, lambda = lambda_input,  scaling = "none");

## Running methods
OUTPUTS = [];  # List of saved outputs
#######

# Most methods are called by name, for example the SVRG
output= minimizeFunc_grid_stepsize(prob, "SVRG", options);
OUTPUTS = [OUTPUTS ; output];

# or the "SVRG2" type methods from [1]
# options.embeddim = 10;
# output= minimizeFunc_grid_stepsize(prob, "AMprev", options);
# OUTPUTS = [OUTPUTS ; output];

# The SAGA methods need to be initited before by calling "initiate_SAGA" for example
options.stepsize_multiplier =10;
sg = initiate_SAGA(prob , options, minibatch_type = "partition", probability_type= "opt")
output= minimizeFunc(prob, sg, options);
OUTPUTS = [OUTPUTS ; output];
#######

# There are also several full batch methods such as gradient descent
method_name = "BFGS";
output1= minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS ; output1];

# and even an accelerated BFGS method
options.embeddim = [prob.numdata, 1/prob.numfeatures]; # Acceleration parameters
method_name = "BFGS_accel";
output3= minimizeFunc_grid_stepsize(prob, method_name, options);
OUTPUTS = [OUTPUTS ; output3];

# saving the data for later
default_path = "./data/";   savename= replace(replace(prob.name, r"[\/]", "-"),".","_");
save("$(default_path)$(savename).jld", "OUTPUTS",OUTPUTS);

#plot and save graphs
pyplot()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS,prob,options) # Plot and save output

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
