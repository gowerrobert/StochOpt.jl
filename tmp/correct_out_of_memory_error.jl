# println("Free memory: ", Sys.free_memory() / 2^20, " MiB")
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

@time include("./src/StochOpt.jl") # Be carefull about the path here

default_path = "./data/";

Random.seed!(1);

### LOADING DATA ###
println("--- Loading data ---");
datasets = readlines("$(default_path)available_datasets.txt");

## Only loading datasets, no data generation
# idx = 4; # australian
idx = 14; # news20.binary
data = datasets[idx];

@time X, y = loadDataset(data);

# varinfo(r"(X|y|prob)")

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-1), max_iter=10^8, max_time=10.0^2, max_epocs=10^8,
                    #   regularizor_parameter = "1/num_data", # fixes lambda
                      regularizor_parameter = "normalized",
                    #   regularizor_parameter = "Lbar/n",
                    #   repeat_stepsize_calculation=true, # used in minimizeFunc_grid_stepsize
                      initial_point="zeros", # is fixed not to add more randomness 
                      force_continue=false); # force continue if diverging or if tolerance reached

@time prob = load_logistic_from_matrices(X, y, data, options, lambda=-1, scaling="none");  # scaling = centering and scaling

get_fsol_logistic!(prob)


# # varinfo(r"(X|y|prob)")

# X = nothing; # available in prob.X
# y = nothing; # available in prob.y
# # varinfo(r"(X|y|prob)")

# ### Trying to run SVRG to compute prob.fsol
# method_input = "BFGS";
# options = set_options(tol=10.0^(-6.0), skip_error_calculation=10^5, exacterror=false, max_iter=10^8, 
#                       max_time=60.0*60.0*3.0, max_epocs=10^8, repeat_stepsize_calculation=true, rep_number=2);
# # options.batchsize = 1;
# options.batchsize = prob.numdata;
# output = minimizeFunc(prob, method_input, options);
# # output = minimizeFunc_grid_stepsize(prob, method_input, options);

# ## Ploting the two solutions
# gr()# gr() pyplot() # pgfplots() #plotly()
# plot_outputs_Plots([output], prob, options);

# fsol = minimum([output.fs]);

# fsolfilename = get_fsol_filename(prob);
# save("$(fsolfilename).jld", "fsol", fsol)