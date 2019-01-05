## EXPERIMENT 4

## Testing the optimality of our tau* for the same gamma = gamma_heuristic
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

Random.seed!(1);

### LOADING THE DATA ###
println("--- Loading data ---");
default_path = "./data/";
datasets = readlines("$(default_path)available_datasets.txt");
idx = 3; # YearPredictionMSD
data = datasets[idx];
X, y = loadDataset(data);

######################################## SETTING UP THE PROBLEM ########################################
println("\n--- Setting up the selected problem ---");
# scaling = "none";
scaling = "column-scaling";
options = set_options(tol=10.0^(-1), max_iter=10^8, max_epocs=10^8,
                      max_time=60.0,
                      skip_error_calculation=10^4,
                      batchsize=1,
                      regularizor_parameter = "normalized",
                      initial_point="zeros", # is fixed not to add more randomness
                      force_continue=false); # force continue if diverging or if tolerance reached
u = unique(y);
if length(u) < 2
    error("Wrong number of possible outputs")
elseif length(u) == 2
    println("Binary output detected: the problem is set to logistic regression")
    prob = load_logistic_from_matrices(X, y, data, options, lambda=-1, scaling=scaling);
else
    println("More than three modalities in the outputs: the problem is set to ridge regression")
    prob = load_ridge_regression(X, y, data, options, lambda=-1, scaling=scaling); #column-scaling
end

n = prob.numdata;
d = prob.numfeatures;
mu = prob.mu;
Lmax = prob.Lmax;
L = prob.L;
Lbar = prob.Lbar;

if occursin("lgstc", prob.name) # julia 0.7
    ## Correcting for logistic since phi'' <= 1/4
    Lmax /= 4;
end

### Plot of the empirical complexity vs tau (averaged runs) ###
tau_simple = round(Int, 1 + (mu*(n-1))/(4*Lbar));
tau_bernstein = max(1, round(Int, 1 + (mu*(n-1))/(8*L) - (4/3)*log(d)*((n-1)/n)*(Lmax/(2*L))));
tau_heuristic = round(Int, 1 + (mu*(n-1))/(4*L));

######################################## EMPIRICAL OPTIMAL MINIBATCH SIZE ########################################
## Empirical stepsizes returned by optimal mini-batch SAGA with line searchs
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
minibatchlist = [1, 2, 3, 5, 10, 20, 50];
# minibatchlist = [10, 50, 100, 1000];


# minibatchlist = [100];
# minibatchlist = [1, 10, 50];
# minibatchlist = [50, 10, 1];

# minibatchlist = [1];
# minibatchlist = [5, 1];
# minibatchlist = 5:-1:1;
# minibatchlist = [1];

numsimu = 5; # number of runs of mini-batch SAGA for averaging the empirical complexity
@time OUTPUTS, itercomplex = simulate_SAGA_nice(prob, minibatchlist, options, numsimu);

## Checking that all simulations reached tolerance
fails = [OUTPUTS[i].fail for i=1:length(minibatchlist)*numsimu];
if all(s->(string(s)=="tol-reached"), fails)
    println("Tolerance always reached")
end

## Plotting one SAGA-nice simulation for each mini-batch size
if(numsimu==1)
    gr()
    # pyplot()
    plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp4.1"); # Plot and save output
end

## Computing the empirical complexity
# itercomplex -= 1; #-> should we remove 1 from itercomplex?
empcomplex = reshape(minibatchlist.*itercomplex, length(minibatchlist)); # tau times number of iterations
min_empcomplex, idx_min = findmin(empcomplex);
tau_emp = minibatchlist[idx_min];

pyplot()
plot_empirical_complexity(prob, minibatchlist, empcomplex, 
                          tau_simple, tau_bernstein, 
                          tau_heuristic, tau_emp);
println("\nSimple optimal tau = ", tau_simple);
println("Bernstein optimal tau = ", tau_bernstein);
println("Heuristic optimal tau = ", tau_heuristic);
println("The empirical optimal tau = ", tau_emp);


### Plot of of run for which our tau* is faster ###