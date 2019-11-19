### EXPERIMENT 3

## Comparing different classical settings of SAGA and ours

using JLD
using Plots
using StatsBase
using Match
using Combinatorics
using Random
using Printf
using LinearAlgebra
using Statistics
using Base64
using Formatting
using LaTeXStrings

## Manual inputs
include("./src/StochOpt.jl") # Be carefull about the path here
default_path = "./data/";

# filename = "ridge_YearPredictionMSD_full-column-scaling-regularizor-1e-01-exp3_2"
# data = "YearPredictionMSD_full"
# scaling = "column-scaling";
# lambda = 10^(-1);

# filename = "lgstc_covtype_binary-column-scaling-regularizor-1e-03-exp3_2"
# data = "covtype_binary"
# scaling = "column-scaling";
# lambda = 10^(-3);

filename = "lgstc_real-sim-none-regularizor-1e-01-exp3"
data = "real-sim"
scaling = "none";
lambda = 10^(-1);

numsimu = 1;

Random.seed!(1);

### LOADING THE DATA ###
println("--- Loading data ---");
X, y = loadDataset(default_path, data);

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-4), max_iter=10^8, max_epocs=10^5, # for slice scaled 1e-1
                      max_time=60.0*60.0,
                      skip_error_calculation=10^5,
                      batchsize=1,
                      regularizor_parameter = "normalized",
                      initial_point="zeros", # is fixed not to add more randomness
                      force_continue=true); # force continue if diverging or if tolerance reached
u = unique(y);
if length(u) < 2
    error("Wrong number of possible outputs");
elseif length(u) == 2
    println("Binary output detected: the problem is set to logistic regression")
    prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling=scaling);
else
    println("More than three modalities in the outputs: the problem is set to ridge regression")
    prob = load_ridge_regression(X, y, data, options, lambda=lambda, scaling=scaling); #column-scaling
end

X = nothing;
y = nothing;

n = prob.numdata;
d = prob.numfeatures;
mu = prob.mu
Lmax = prob.Lmax;
L = prob.L;


### II) tau = tau* ###
# Hofmann : tau = 20, gamma = gamma(20)
##---------- Computing step sizes ----------
tau_defazio = 1;
step_defazio = 1.0 / (3.0*(Lmax + n*mu));

tau_hofmann = 20;
K = (4.0*tau_hofmann*Lmax) / (n*mu);
step_hofmann = K / (2*Lmax*(1+K+sqrt(1+K^2)));
# step_hofmann = tau/(mu*n);

# rho = ( n*(n - tau_hofmann) ) / ( tau_hofmann*(n-1) ); # Sketch residual rho = n*(n-b)/(b*(n-1)) in JacSketch paper, page 35
rightterm = ( Lmax*(n - tau_hofmann) ) / ( tau_hofmann*(n-1) ) + ( (mu*n) / (4*tau_hofmann) ); # Right-hand side term in the max
heuristicbound = ( n*(tau_hofmann-1)*L + (n-tau_hofmann)*Lmax ) / ( tau_hofmann*(n-1) );
step_hofmann_heuristic = 0.25 / max(heuristicbound, rightterm);

## Is our optimal tau always one???
## YearPredictionMSD scaled + mu = 10^(-3) => 13
## YearPredictionMSD scaled + mu = 10^(-1) => 1245
tau_heuristic = round(Int, 1 + ( mu*(n-1) ) / ( 4*L ) );
# tau_heuristic = 20;
# rho = ( n*(n - tau_heuristic) ) / ( tau_heuristic*(n-1) ); # Sketch residual rho = n*(n-b)/(b*(n-1)) in JacSketch paper, page 35
rightterm = ( Lmax*(n - tau_heuristic) ) / ( tau_heuristic*(n-1) ) + ( (mu*n) / (4*tau_heuristic) ); # Right-hand side term in the max
heuristicbound = ( n*(tau_heuristic-1)*L + (n-tau_heuristic)*Lmax ) / ( tau_heuristic*(n-1) );
step_heuristic = 0.25 / max(heuristicbound, rightterm);

options.batchsize = tau_heuristic;
if options.batchsize == 1
    method_name = "SAGA-nice";
elseif options.batchsize > 1
    method_name = string("SAGA-", options.batchsize, "-nice");
else
    error("Invalid batch size");
end

step_heuristic_gridsearch, = get_saved_stepsize(prob.name, method_name, options);

str_step_defazio = @sprintf "%.2e" step_defazio
str_step_heuristic = @sprintf "%.2e" step_heuristic
str_step_heuristic_gridsearch = @sprintf "%.2e" step_heuristic_gridsearch
str_step_hofmann = @sprintf "%.2e" step_hofmann
method_names = [latexstring("\$b_\\mathrm{Defazio} \\; \\; = 1 \\ \\ + \\gamma_\\mathrm{Defazio} \\ \\ \\: \\: = $str_step_defazio\$"),
                latexstring("\$b_\\mathrm{practical} \\, = $tau_heuristic + \\gamma_\\mathrm{practical} \\ \\ = $str_step_heuristic\$"),
                latexstring("\$b_\\mathrm{practical} \\, = $tau_heuristic + \\gamma_\\mathrm{grid search} = $str_step_heuristic_gridsearch\$"),
                latexstring("\$b_\\mathrm{Hofmann} = 20 + \\gamma_\\mathrm{Hofmann}  \\ \\, = $str_step_hofmann\$")];
mini_batch_sizes = [tau_defazio, tau_heuristic, tau_heuristic, tau_hofmann];
stepsizes = [step_defazio, step_heuristic, step_heuristic_gridsearch, step_hofmann];


## LOAD DATA
OUTPUTS = load("$(default_path)$(filename).jld", "OUTPUTS")

## "ridge_YearPredictionMSD_full-column-scaling-regularizor-1e-01-exp3_2"
# nbpoints = 60;
# freq = round(Int, length(OUTPUTS[4].fs)/nbpoints)
# idx = collect(1:freq:length(OUTPUTS[4].fs))
# OUTPUTS[4].fs = OUTPUTS[4].fs[idx];

## "lgstc_covtype_binary-column-scaling-regularizor-1e-03-exp3_2"
# freq = 4;
# length(OUTPUTS[1].fs)
# idx1 = collect(1:freq:length(OUTPUTS[1].fs))
# OUTPUTS[1].fs = OUTPUTS[1].fs[idx1];
# freq = 8;
# length(OUTPUTS[2].fs)
# idx2 = collect(1:freq:length(OUTPUTS[2].fs))
# OUTPUTS[2].fs = OUTPUTS[2].fs[idx2];
# freq = 1;
# length(OUTPUTS[3].fs)
# idx3 = collect(1:freq:length(OUTPUTS[3].fs))
# OUTPUTS[3].fs = OUTPUTS[3].fs[idx3];
# freq = 2;
# length(OUTPUTS[4].fs)
# idx4 = collect(1:freq:length(OUTPUTS[4].fs))
# OUTPUTS[4].fs = OUTPUTS[4].fs[idx4];

## Re-plotting one SAGA-nice simulation for each mini-batch size
pyplot()
plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp3-cosmit", legendpos=:topright); # Plot and save output

OUTPUTS_without_hofmann = OUTPUTS[1:3];
# OUTPUTS_without_hofmann = [OUTPUTS_without_hofmann; OUTPUTS[3:5]];
pyplot()
plot_outputs_Plots(OUTPUTS_without_hofmann, prob, options, suffix="_without_hofmann-exp3-cosmit"); # Plot and save output

