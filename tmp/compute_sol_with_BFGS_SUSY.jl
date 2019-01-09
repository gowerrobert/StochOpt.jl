using JLD
using Plots
using StatsBase
using Match
using Combinatorics
using Random # julia 0.7
using Printf # julia 0.7
using LinearAlgebra # julia 0.7
using Statistics # julia 0.7
using Base64 # julia 0.7

include("../src/StochOpt.jl") # Be carefull about the path here

default_path = "./data/";

Random.seed!(1);

### LOADING DATA ###
println("--- Loading data ---");
datasets = readlines("$(default_path)available_datasets.txt");

## Only loading datasets, no data generation
idx = 13; # SUSY
data = datasets[idx];

@time X, y = loadDataset(default_path, data);

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-1), max_iter=10^8, max_time=10.0^2, max_epocs=10^8,
                    #   regularizor_parameter = "1/num_data", # fixes lambda
                      regularizor_parameter = "normalized",
                    #   regularizor_parameter = "Lbar/n",
                      initial_point="zeros", # is fixed not to add more randomness 
                      force_continue=false); # force continue if diverging or if tolerance reached

@time prob = load_logistic_from_matrices(X, y, data, options, lambda=-1, scaling="column-scaling");

X = nothing; # available in prob.X
y = nothing; # available in prob.y

########################################### SUSY ############################################
#region
## Computing the solution with a serial gridsearch
@time get_fsol_logistic!(prob)
## ---> First gridsearch on SUSY-none of 2 hours led us to fsol = 4.530221721802407e-6
## (for BFGS, beststep = 2^5.0 and batchsize = prob.numdata)
## (for BFGS-a-5.0e6-0.01, beststep = 2^5.0 and batchsize = prob.numdata)

## ---> First gridsearch on SUSY-column-scaling of ... hours led us to fsol = ...
## (for BFGS, beststep = ... and batchsize = prob.numdata)
## (for BFGS-a ..., beststep = ... and batchsize = prob.numdata)


# ## By hand, let us try another single run of 3 hours with step = ... on SUSY
# # method_input = "SVRG";

# # options = set_options(tol=10.0^(-16.0), skip_error_calculation=10^1, exacterror=false, max_iter=10^8, 
# #                       max_time=60.0*60.0*3.0, max_epocs=70, force_continue=true);
# # options.batchsize = 100;
# # options.stepsize_multiplier = 2.0; # beststep = 2.0 for batchsize = 100

# # ## Saving the optimization output in a JLD file
# # output = minimizeFunc(prob, method_input, options);

# # ## Saving the optimization output in a JLD file
# # a, savename = get_saved_stepsize(prob.name, method_input, options);
# # a = nothing;
# # save("$(default_path)$(savename)_try_2.jld", "output", output)

# # ## Setting the true solution as the smallest of both
# # prob.fsol = minimum(output.fs);
# # println("\n----------------------------------------------------------------------")
# # @printf "For %s, fsol = %f\n" prob.name prob.fsol
# # println("----------------------------------------------------------------------\n")

# # ## Saving the solution in a JLD file
# # fsolfilename = get_fsol_filename(prob); # not coherent with get_saved_stepsize output
# # save("$(fsolfilename)_try_2.jld", "fsol", prob.fsol)
#endregion
######################################################################################################
