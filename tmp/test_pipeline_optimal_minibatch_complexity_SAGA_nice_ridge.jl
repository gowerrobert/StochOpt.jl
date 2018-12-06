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

include("./src/StochOpt.jl") # Be carefull about the path here

Random.seed!(1234);

### LOADING DATA ###
println("--- Loading data ---");
# Available datasets are in "./data/available_datasets.txt" 
datasets = ["gauss-5-8-0.0_seed-1234", "YearPredictionMSD", "abalone", "housing"];
#, "letter_scale", "heart", "phishing", "madelon", "a9a",
# "mushrooms", "phishing", "w8a", "gisette_scale",

## Only loading datasets, no generation
data = datasets[1];
try
    X, y = loadDataset(data);
catch loaderror
    println(loaderror);
    println("Check the list of available datasets in: ./data/available_datasets.txt");
end

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the ridge regression problem ---");
options = set_options(max_iter=10^8, max_time=10.0, max_epocs=1000, repeat_stepsize_calculation=true, skip_error_calculation=51,
                      force_continue=true, initial_point="randn", batchsize=0);
prob = load_ridge_regression(X, y, probname, options, lambda=-1, scaling="none");  # Disabling scaling
# QUESTION: how is lambda selected?
n = prob.numdata;
d = prob.numfeatures;

### COMPUTING THE SMOOTHNESS CONSTANTS ###
# Compute the smoothness constants L, L_max, \cL, \bar{L}
datathreshold = 24; # if n is too large we do not compute the exact expected smoothness constant nor its relative quantities

println("\n--- Computing smoothness constants ---");
mu = get_mu_str_conv(prob); # mu = minimum(sum(prob.X.^2, 1)) + prob.lambda;
L = get_LC(prob, collect(1:n)); # L = eigmax(prob.X*prob.X')/n + prob.lambda;
Li_s = get_Li(prob);
Lmax = maximum(Li_s); # Lmax = maximum(sum(prob.X.^2, 1)) + prob.lambda;
Lbar = mean(Li_s);


########################### EMPIRICAL UPPER BOUNDS OF THE EXPECTED SMOOTHNESS CONSTANT ###########################
#region
### COMPUTING THE BOUNDS ###
simplebound, bernsteinbound, heuristicbound, expsmoothcst = get_expected_smoothness_bounds(prob);

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
simplestepsize, bernsteinstepsize, heuristicstepsize, expsmoothstepsize = get_stepsize_bounds(prob, simplebound, bernsteinbound, heuristicbound, expsmoothcst);

### PLOTING ###
println("\n--- Ploting stepsizes ---");
# PROBLEM: there is still a problem of ticking non integer on the xaxis
pyplot()
plot_stepsize_bounds(prob, simplestepsize, bernsteinstepsize, heuristicstepsize, expsmoothstepsize);
#endregion
##################################################################################################################


###################################### THEORETICAL OPTIMAL MINI-BATCH SIZES ######################################
#region
## Compute optimal mini-batch size
if typeof(expsmoothcst)==Array{Float64,2}
    LHS = 4*(1:n).*(expsmoothcst .+ prob.lambda)./mu;
    RHS = n .+ (n .- (1:n)) .* (4*(Lmax+prob.lambda)/((n-1)*mu));
    exacttotalcplx = reshape(max.(LHS, RHS), n);
    opt_minibatch_exact = argmin(exacttotalcplx);
else
    opt_minibatch_exact = nothing;
end

## WARNING: Verify computations : should we add lambda????
opt_minibatch_simple = round(Int, 1 + (mu*(n-1))/(4*Lbar)); # One should not add again lambda since it is already taken into account in Lbar
opt_minibatch_bernstein = max(1, round(Int, 1 + (mu*(n-1))/(8*L) - (4/3)*log(d)*((n-1)/n)*(Lmax/(2*L)) )); ## WARNING: Verify computations : should we add lambda????
opt_minibatch_heuristic = round(Int, 1 + (mu*(n-1))/(4*L));
#endregion
##################################################################################################################


########################################### SAVNG RESULTS ########################################################
save_SAGA_nice_constants(prob, data, simplebound, bernsteinbound, heuristicbound, expsmoothcst, 
                         simplestepsize, bernsteinstepsize, heuristicstepsize, expsmoothstepsize,
                         opt_minibatch_simple, opt_minibatch_bernstein, opt_minibatch_heuristic, 
                         opt_minibatch_exact);
##################################################################################################################


# ######################################## EMPIRICAL OPTIMAL MINIBATCH SIZE ########################################
# ## Empirical stepsizes returned by optimal mini-batch SAGa with line searchs
# # if(n <= datathreshold)
# #     minibatchlist = 1:n;
# # elseif(opt_minibatch_simple>2)
# #     minibatchlist = [1; opt_minibatch_simple; opt_minibatch_heuristic; round(Int, (opt_minibatch_simple+n)/2); round(Int, sqrt(n)); n]#[collect(1:(opt_minibatch_simple+1)); n];
# # else
# #     minibatchlist = [collect(1:(opt_minibatch_heuristic+1)); round(Int, sqrt(n)); n];
# # end

# ## For abalone dataset
# # minibatchlist = [collect(1:10); 50; 100];
# # minibatchlist = collect(1:8);

# ## For n=24
# # minibatchlist = [collect(1:6); 12; 24];

# ## For n=5000
# # minibatchlist = [1; 5; 10; 50; 100; 200; 1000; 5000];

# ## For n=500
# # minibatchlist = collect(1:10);

# ## For YearPredictionMSD
# # minibatchlist = [1, 2, 3, 5];
# # minibatchlist = [1, 2, 3, 5, 10, 20, 50];
# # minibatchlist = [10, 50, 100, 1000];


# # minibatchlist = [100];
# # minibatchlist = [1, 10, 50];
# # minibatchlist = [50, 10, 1];

# # minibatchlist = [1];
# minibatchlist = [5, 1];
# # minibatchlist = 5:-1:1;
# # minibatchlist = [1];


# # srand(1234);

# numsimu = 5; # number of runs of mini-batch SAGA for averaging the empirical complexity

# tic();
# OUTPUTS, itercomplex = simulate_SAGA_nice(prob, minibatchlist, numsimu, tolerance=10.0^(-1));
# toc();

# ## Checking that all simulations reached tolerance
# fails = [OUTPUTS[i].fail for i=1:length(minibatchlist)*numsimu];
# if all(s->(string(s)=="tol-reached"), fails)
#     println("Tolerance always reached")
# end

# ## Plotting one SAGA-nice simulation for each mini-batch size
# if(numsimu==1)
#     gr()
#     # pyplot()
#     plot_outputs_Plots(OUTPUTS, prob, options); # Plot and save output
# end

# ## Computing the empirical complexity
# # itercomplex -= 1; #-> should we remove 1 from itercomplex?
# empcomplex = reshape(minibatchlist.*itercomplex, length(minibatchlist)); # tau times number of iterations
# opt_minibatch_emp = minibatchlist[indmin(empcomplex)];

# pyplot()
# plot_empirical_complexity(prob, minibatchlist, empcomplex, 
#                           opt_minibatch_simple, opt_minibatch_bernstein, 
#                           opt_minibatch_heuristic, opt_minibatch_emp);


# ######################################### PRINTING CONSTANTS AND RESULTS #########################################
# println("\nPROBLEM DIMENSIONS:");
# println("   Number of datapoints = ", n); # n in the paper notation
# println("   Number of features = ", d); # d in the paper notation

# println("\nSimple optimal tau = ", opt_minibatch_simple);
# println("Bernstein optimal tau = ", opt_minibatch_bernstein);
# println("Heuristic optimal tau = ", opt_minibatch_heuristic);
# println("The empirical optimal tau = ", opt_minibatch_emp);

# # println("List of mini-batch sizes = ", minibatchlist);
# println("\nEmpirical complexity = ", empcomplex);

# # println("\nSMOOTHNESS CONSTANTS:");
# # println("   Lmax = ", Lmax);
# # println("   L = ", L);
# # println("Li_s = ", Li_s);
# # println("   Lbar = ", Lbar);
# ##################################################################################################################