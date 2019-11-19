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

## Bash inputs
# include("../src/StochOpt.jl") # Be carefull about the path here
# default_path = "./data/";
# data = ARGS[1];
# scaling = ARGS[2];
# lambda = parse(Float64, ARGS[3]);
# stepsize_multiplier = parse(Float64, ARGS[4]);
# println("Inputs: ", data, " + ", scaling, " + ", lambda, " + stepsize_multiplier = ",  stepsize_multiplier, "\n");

## Manual inputs
include("./src/StochOpt.jl") # Be carefull about the path here
default_path = "./data/";
# datasets = readlines("$(default_path)available_datasets.txt");
# idx = 11;
# data = datasets[idx];
data = "real-sim"
scaling = "none";
# scaling = "column-scaling";
# lambda = -1;
lambda = 10^(-3);
# lambda = 10^(-1);
stepsize_multiplier = 2^(-1.0);

Random.seed!(1);

### LOADING DATA ###
println("--- Loading data ---");
@time X, y = loadDataset(default_path, data);

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-16.0), skip_error_calculation=1,
                      exacterror=false, max_iter=10^8,
                      max_time=60.0*60.0, max_epocs=2000, force_continue=true);
@time prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling=scaling);


######################################################################################
#region
## Computing the solution with a serial gridsearch
# @time get_fsol_logistic!(prob)

options = set_options(tol=10.0^(-16.0), skip_error_calculation=1, exacterror=false, max_iter=10^8,
                              max_time=60.0, max_epocs=10^5, repeat_stepsize_calculation=true, rep_number=2);
## Running BFGS
options.batchsize = prob.numdata;
method_input = "BFGS";
# grid = [2.0^(9), 2.0^(7), 2.0^(5), 2.0^(3), 2.0^(1), 2.0^(-1), 2.0^(-3), 2.0^(-5)];
# output = minimizeFunc_grid_stepsize(prob, method_input, options, grid=grid);
output = minimizeFunc_grid_stepsize(prob, method_input, options);

## Result = stepsize = 0.5 for none and both lambdas
## Result = stepsize =  for column-scaling

## Result = stepsize = 0.125 for none and lambda = 0.001

## Running BFGS
stepsize_multiplier = 0.05
options = set_options(tol=10.0^(-16.0), skip_error_calculation=10,
                      exacterror=false, max_iter=10^8,
                      max_time=60.0*60.0*12, # 12 hours
                      max_epocs=1000,
                      force_continue=true);
options.stepsize_multiplier = stepsize_multiplier;
options.batchsize = prob.numdata;
method_input = "BFGS";
output = minimizeFunc(prob, method_input, options);

## Saving the optimization output in a JLD file
a, savename = get_saved_stepsize(prob.name, method_input, options);
a = nothing;
save("$(default_path)$(savename).jld", "output", output)

## Setting the true solution as the smallest of both
prob.fsol = minimum(output.fs[.!isnan.(output.fs)]);
println("\n----------------------------------------------------------------------")
@printf "For %s, fsol = %1.50f\n" prob.name prob.fsol
println("----------------------------------------------------------------------\n")

## Saving the solution in a JLD file
fsolfilename = get_fsol_filename(prob); # not coherent with get_saved_stepsize output
save("$(fsolfilename).jld", "fsol", prob.fsol);
#endregion
######################################################################################################

n = prob.numdata;
d = prob.numfeatures;
mu = prob.mu
Lmax = prob.Lmax;
L = prob.L;
# Lbar = prob.Lbar;

options = set_options(max_iter=10^8, max_time=10.0^(5.0), max_epocs=8, force_continue=true, initial_point="zeros",
                      exacterror=false);

## First test of SAGA-nice (hand-made settings)
options.stepsize_multiplier = 1e-2;
options.batchsize = 1;
options.skip_error_calculation = 5000;
SAGA_nice = initiate_SAGA_nice(prob, options); # separated implementation from SAGA
output = minimizeFunc(prob, SAGA_nice, options);

## Test with theoretical parameters
b_practical = round(Int, 1 + (mu*(n-1))/(4*L))
# rho = ( n*(n - b_practical) ) / ( b_practical*(n-1) ); # Sketch residual rho = n*(n-b)/(b*(n-1)) in JacSketch paper, page 35
rightterm = ( Lmax*(n - b_practical) ) / ( b_practical*(n-1) ) + ( (mu*n) / (4*b_practical) ); # Right-hand side term in the max
practical_bound = ( n*(b_practical-1)*L + (n-b_practical)*Lmax ) / ( b_practical*(n-1) );
step_practical = 0.25 / max(practical_bound, rightterm);

options.stepsize_multiplier = step_practical
options.batchsize = b_practical
options.skip_error_calculation = 1;
SAGA_nice = initiate_SAGA_nice(prob, options); # separated implementation from SAGA
output = minimizeFunc(prob, SAGA_nice, options);


## Test in relative error
@time X, y = loadDataset(default_path, data);

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(max_iter=10^8, max_time=10.0^(5.0), max_epocs=8, force_continue=true, initial_point="zeros");
@time prob = load_logistic_from_matrices(X, y, data, options, lambda=lambda, scaling=scaling);

options.stepsize_multiplier = step_practical
options.batchsize = b_practical
options.skip_error_calculation = 1;
SAGA_nice = initiate_SAGA_nice(prob, options); # separated implementation from SAGA
output = minimizeFunc(prob, SAGA_nice, options);


# Last time (09/10/2019) BFGS was launched for 12 hours
stepsize_multiplier = 0.05
options = set_options(tol=10.0^(-16.0), skip_error_calculation=10,
                      exacterror=false, max_iter=10^8,
                      max_time=60.0*60.0*12, # 12 hours
                      max_epocs=1000,
                      force_continue=true);
options.stepsize_multiplier = 0.05;
options.batchsize = prob.numdata;
method_input = "BFGS";
output = minimizeFunc(prob, method_input, options);

# Skipping 10 iterations per epoch
# BFGS
# ----------------------------------------------------------------------
#       It    |                 f(x)          |   epochs  |     Time   |
# ----------------------------------------------------------------------
# Skipping 10 iterations per epoch
#         10  |           0.51749126461847527736           |    10.00  |  625.8210  |
#         20  |           0.51155904932584372879           |    20.00  |  1256.2888  |
#         30  |           0.50655372751985006108           |    30.00  |  1799.3795  |
#         40  |           0.49355427226909964755           |    40.00  |  2281.8571  |
#         50  |           0.49209853591370084080           |    50.00  |  2815.8171  |
#         60  |           0.49202543516995400630           |    60.00  |  3329.7424  |
#         70  |           0.49157084448861293469           |    70.00  |  3668.4253  |
#         80  |           0.49104274596500657735           |    80.00  |  3949.6528  |
#         90  |           0.49101780512170817294           |    90.00  |  4230.9330  |
#        100  |           0.49101685005023137931           |   100.00  |  4517.0339  |
#        110  |           0.49101118228138929123           |   110.00  |  4822.1032  |
#        120  |           0.49100296642455520946           |   120.00  |  5104.8637  |
#        130  |           0.49100249501295328836           |   130.00  |  5396.5249  |
#        140  |           0.49100248291152076563           |   140.00  |  5678.0957  |
#        150  |           0.49100243538335752724           |   150.00  |  5972.5602  |
#        160  |           0.49100233904208556712           |   160.00  |  6290.8388  |
#        170  |           0.49100233136014853619           |   170.00  |  6566.9251  |
#        180  |           0.49100233114468905260           |   180.00  |  7037.6969  |
#        190  |           0.49100233028106610167           |   190.00  |  7590.0722  |
#        200  |           0.49100232873419186186           |   200.00  |  8088.5499  |
#        210  |           0.49100232862355952523           |   210.00  |  8667.0835  |
#        220  |           0.49100232861725251476           |   220.00  |  10259.9902  |
#        230  |           0.49100232857636949557           |   230.00  |  11942.3620  |
#        240  |           0.49100232853593139826           |   240.00  |  13627.4415  |
#        250  |           0.49100232853423525503           |   250.00  |  15164.8171  |
#        260  |           0.49100232853416075907           |   260.00  |  16770.4709  |
#        270  |           0.49100232853368935837           |   270.00  |  18415.5585  |
#        280  |           0.49100232853310621373           |   280.00  |  19963.3772  |
#        290  |           0.49100232853307684833           |   290.00  |  21490.3292  |
#        300  |           0.49100232853307568259           |   300.00  |  23080.3403  |
#        310  |           0.49100232853306857717           |   310.00  |  24707.2328  |
#        320  |           0.49100232853305914027           |   320.00  |  26296.6824  |
#        330  |           0.49100232853305864067           |   330.00  |  27815.3988  |
#        340  |           0.49100232853305858516           |   340.00  |  29527.6967  |
#        350  |           0.49100232853305864067           |   350.00  |  31056.4852  |
#        360  |           0.49100232853305847414           |   360.00  |  32716.7698  |
#        370  |           0.49100232853305847414           |   370.00  |  34407.6138  |
#        380  |           0.49100232853305847414           |   380.00  |  36157.6337  |
#        390  |           0.49100232853305841862           |   390.00  |  37705.8482  |
#        400  |           0.49100232853305841862           |   400.00  |  39305.7948  |
#        410  |           0.49100232853305847414           |   410.00  |  40777.0965  |
#        420  |           0.49100232853305847414           |   420.00  |  42382.5897  |
#        426  |           0.49100232853305847414           |   426.00  |  43211.9868  |
