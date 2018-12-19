using JLD
using Plots
using Printf # julia 0.7
using Match
using LinearAlgebra # julia 0.7
using Random # julia 0.7
using Combinatorics
using StatsBase

# using Statistics # julia 0.7
# using Base64 # julia 0.7

include("./src/StochOpt.jl") # Be carefull about the path here

default_path = "./data/";

Random.seed!(1);

### LOADING DATA ###
println("--- Loading data ---");
# Available datasets are in "./data/available_datasets.txt" 
# datasets = ["fff", "gauss-5-8-0.0_seed-1234", "YearPredictionMSD", "abalone", "housing"];
datasets = readlines("$(default_path)available_datasets.txt");
for i in 1:length(datasets) println(i, ": ", datasets[i]) end

## Only loading datasets, no data generation
data = datasets[1];

X, y = loadDataset(data);

scaling_rule = "none";
# scaling_rule = "column-scaling";

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-3), max_iter=10^8, max_time=10.0^2, max_epocs=10^8,
                    #   regularizor_parameter = "1/num_data", # fixes lambda
                      regularizor_parameter = "normalized",
                    #   regularizor_parameter = "Lbar/n",
                    #   repeat_stepsize_calculation=true, # used in minimizeFunc_grid_stepsize
                      initial_point="zeros", # is fixed not to add more randomness 
                      force_continue=false); # force continue if diverging or if tolerance reached
u = unique(y);
if length(u) < 2
    error("Wrong number of possible outputs")
elseif length(u) == 2
    println("Binary output detected: the problem is set to logistic regression")
    prob = load_logistic_from_matrices(X, y, data, options, lambda=-1, scaling=scaling_rule);  # scaling = centering and scaling
else
    println("More than three modalities in the outputs: the problem is set to ridge regression")
    prob = load_ridge_regression(X, y, data, options, lambda=-1, scaling=scaling_rule);
end

n = prob.numdata;
d = prob.numfeatures;

### COMPUTING THE SMOOTHNESS CONSTANTS ###
# Compute the smoothness constants L, L_max, \cL, \bar{L}
datathreshold = 24; # if n is too large we do not compute the exact expected smoothness constant nor its relative quantities

expsmoothcst = nothing;

########################### EMPIRICAL UPPER BOUNDS OF THE EXPECTED SMOOTHNESS CONSTANT ###########################
#region
### COMPUTING THE BOUNDS ###
simplebound, bernsteinbound, heuristicbound, expsmoothcst = get_expected_smoothness_bounds(prob); # WARNING : markers are missing!

### PLOTING ###
println("\n--- Ploting upper bounds ---");
# PROBLEM: there is still a problem of ticking non integer on the xaxis
pyplot()
plot_expected_smoothness_bounds(prob, simplebound, bernsteinbound, heuristicbound, expsmoothcst);

# heuristic equals true expected smoothness constant for tau=1 and n as expected, else it is above as hoped
if(n<=datathreshold)
    println("Heuristic - expected smoothness gap: ", heuristicbound - expsmoothcst)
    println("Simple - heuristic gap: ", simplebound - heuristicbound)
    println("Bernstein - simple gap: ", bernsteinbound - simplebound)
end
#endregion
##################################################################################################################


##################################### EMPIRICAL UPPER BOUNDS OF THE STEPSIZES ####################################
#region
## TO BE DONE: implement grid-search for the stepsizes, i.e.
## 1) set a grid of stepsizes around 1/(4Lmax)
## 2) run several SAGA_nice on the same problem with different stepsize (average?)
## 3) pick the 'best' stepsize

### COMPUTING THE UPPER-BOUNDS OF THE STEPSIZES ###
simplestepsize, bernsteinstepsize, heuristicstepsize, hofmannstepsize, expsmoothstepsize = get_stepsize_bounds(prob, simplebound, bernsteinbound, heuristicbound, expsmoothcst);

### PLOTING ###
println("\n--- Ploting stepsizes ---");
# PROBLEM: there is still a problem of ticking non integer on the xaxis
pyplot()
plot_stepsize_bounds(prob, simplestepsize, bernsteinstepsize, heuristicstepsize, hofmannstepsize, expsmoothstepsize);
#endregion
##################################################################################################################


###################################### THEORETICAL OPTIMAL MINI-BATCH SIZES ######################################
#region
## Compute optimal mini-batch size
if typeof(expsmoothcst)==Array{Float64,2}
    LHS = 4*(1:n).*(expsmoothcst .+ prob.lambda)./prob.mu;
    RHS = n .+ (n .- (1:n)) .* (4*(prob.Lmax+prob.lambda)/((n-1)*prob.mu));
    exacttotalcplx = reshape(max.(LHS, RHS), n);
    opt_minibatch_exact = argmin(exacttotalcplx);
else
    opt_minibatch_exact = nothing;
end

## WARNING: Verify computations : should we add lambda????
opt_minibatch_simple = round(Int, 1 + (prob.mu*(n-1))/(4*prob.Lbar)); # One should not add again lambda since it is already taken into account in Lbar
opt_minibatch_bernstein = max(1, round(Int, 1 + (prob.mu*(n-1))/(8*prob.L) - (4/3)*log(d)*((n-1)/n)*(prob.Lmax/(2*prob.L)) )); ## WARNING: Verify computations : should we add lambda????
opt_minibatch_heuristic = round(Int, 1 + (prob.mu*(n-1))/(4*prob.L));
#endregion
##################################################################################################################


########################################### SAVNG RESULTS ########################################################
#region
save_SAGA_nice_constants(prob, data, simplebound, bernsteinbound, heuristicbound, expsmoothcst, 
                         simplestepsize, bernsteinstepsize, heuristicstepsize, expsmoothstepsize,
                         opt_minibatch_simple, opt_minibatch_bernstein, opt_minibatch_heuristic, 
                         opt_minibatch_exact);
#endregion
##################################################################################################################


######################################## EMPIRICAL OPTIMAL MINIBATCH SIZE ########################################
## Empirical stepsizes returned by optimal mini-batch SAGa with line searchs
# if(n <= datathreshold)
#     minibatchlist = 1:n;
# elseif(opt_minibatch_simple>2)
#     minibatchlist = [1; opt_minibatch_simple; opt_minibatch_heuristic; round(Int, (opt_minibatch_simple+n)/2); round(Int, sqrt(n)); n]#[collect(1:(opt_minibatch_simple+1)); n];
# else
#     minibatchlist = [collect(1:(opt_minibatch_heuristic+1)); round(Int, sqrt(n)); n];
# end

## For abalone dataset
# minibatchlist = [collect(1:10); 50; 100];
# minibatchlist = collect(1:8);

## For n=24
# minibatchlist = [collect(1:6); 12; 24];

## For n=5000
# minibatchlist = [1; 5; 10; 50; 100; 200; 1000; 5000];

## For n=500
# minibatchlist = collect(1:10);

## For YearPredictionMSD
# minibatchlist = [1, 2, 3, 5];
# minibatchlist = [1, 2, 3, 5, 10, 20, 50];
# minibatchlist = [10, 50, 100, 1000];

## For australian (classification)
# minibatchlist = [1, 2, 3, 5, 10];
minibatchlist = [1, 2, 3, 5, 10, 20, 50];


# minibatchlist = 2.^collect(1:10);


# minibatchlist = [1000];
# minibatchlist = [1, 10, 50];
# minibatchlist = [50, 10, 1];

# minibatchlist = [1];
# minibatchlist = [5, 1, n];
# minibatchlist = 5:-1:1;
# minibatchlist = [1];


# srand(1234);

numsimu = 5; # number of runs of mini-batch SAGA for averaging the empirical complexity

minibatchlist = sort(minibatchlist);
@time OUTPUTS, itercomplex = simulate_SAGA_nice(prob, minibatchlist, options, numsimu,
                                                skipped_errors=2000, skip_multiplier=10.0);

## Checking that all simulations reached tolerance
fails = [OUTPUTS[i].fail for i=1:length(minibatchlist)*numsimu];
if all(s->(string(s)=="tol-reached"), fails)
    println("Tolerance always reached");
else
    println("Tolerance not always reached");
end

## Plotting one SAGA-nice simulation for each mini-batch size
if(numsimu==1)
    options.batchsize = 0;
    gr()
    # pyplot()
    plot_outputs_Plots(OUTPUTS, prob, options); # Plot and save output
end

## Computing the empirical complexity
# itercomplex -= 1; #-> should we remove 1 from itercomplex?
empcomplex = reshape(minibatchlist.*itercomplex, length(minibatchlist)); # tau times number of iterations
opt_minibatch_emp = minibatchlist[argmin(empcomplex)]; # julia 0.7

pyplot()
plot_empirical_complexity(prob, minibatchlist, empcomplex, 
                          opt_minibatch_simple, opt_minibatch_bernstein, 
                          opt_minibatch_heuristic, opt_minibatch_emp);

# ######################################### PRINTING CONSTANTS AND RESULTS #########################################
println("\nPROBLEM DIMENSIONS:");
println("   Number of datapoints = ", n); # n in the paper notation
println("   Number of features = ", d); # d in the paper notation

println("\nSimple optimal tau = ", opt_minibatch_simple);
println("Bernstein optimal tau = ", opt_minibatch_bernstein);
println("Heuristic optimal tau = ", opt_minibatch_heuristic);
println("The empirical optimal tau = ", opt_minibatch_emp);

# # println("List of mini-batch sizes = ", minibatchlist);
# println("\nEmpirical complexity = ", empcomplex);

println("\nSMOOTHNESS CONSTANTS:");
println("   mu   = ", prob.mu);
println("   L    = ", prob.L);
println("   Lmax = ", prob.Lmax);
println("   Lbar = ", prob.Lbar);
# ##################################################################################################################