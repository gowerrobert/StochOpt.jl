### EXPERIMENT 3

## Comparing different classical settings of SAGA and ours

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
using Formatting
using LaTeXStrings

## for the skip_error parameter:
#region
"""
    closest_power_of_ten(integer::Int64)

    Compute the closest power of ten of an integer.

#INPUTS:\\
    - **Int64** integer: integer\\
#OUTPUTS:\\
    - **Int64** or **Float64** closest_power: closest power of ten of the input

# Examples
```jldoctest
julia> closest_power(0)
1
julia> closest_power(9)
1
julia> closest_power(204)
100
```
"""
function closest_power_of_ten(integer::Int64)
    if integer < 0
        closest_power = 10.0 ^ (1 - length(string(integer)));
    else
        closest_power = 10 ^ (length(string(integer)) - 1);
    end
    return closest_power
end
#endregion


## Bash inputs
include("../src/StochOpt.jl") # Be carefull about the path here
default_path = "./data/";
data = ARGS[1];
scaling = ARGS[2];
lambda = parse(Float64, ARGS[3]);
skip1 = parse(Int64, ARGS[4]);
skip2 = parse(Int64, ARGS[5]);
skip3 = parse(Int64, ARGS[6]);
skip4 = parse(Int64, ARGS[7]);
skip_error = [skip1 skip2 skip3 skip4];
relaunch_gridsearch = parse(Bool, ARGS[8]);
# skip_mult = parse(Int64, ARGS[4]);
numsimu = 1;
println("Inputs: ", data, " + ", scaling, " + ", lambda, " + numsimu = ",  numsimu, "\n");

## Manual inputs
# include("./src/StochOpt.jl") # Be carefull about the path here
# default_path = "./data/";
# datasets = readlines("$(default_path)available_datasets.txt");
# idx = 12;
# data = datasets[idx];
# # scaling = "none";
# scaling = "column-scaling";
# # lambda = -1;
# # lambda = 10^(-3);
# lambda = 10^(-1);
# numsimu = 1;
# skip_error = [0 0 0 0];
# # skip_error = [10^6 10^4 10^4 10^6]
# skip_mult=1;
# relaunch_gridsearch = false;

println("Relaunch grid-search: ", relaunch_gridsearch);
println("---------------------------------- SKIP_ERROR ------------------------------------------");
println(skip_error);
println("----------------------------------------------------------------------------------------");

Random.seed!(1);

### LOADING THE DATA ###
println("--- Loading data ---");
X, y = loadDataset(default_path, data);

### SETTING UP THE PROBLEM ###
println("\n--- Setting up the selected problem ---");
options = set_options(tol=10.0^(-4), max_iter=10^8, max_epocs=200, # for slice scaled 1e-1
                      max_time=60.0*60.0*5.0,
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

if occursin("lgstc", prob.name) # julia 0.7
    ## Correcting for logistic since phi'' <= 1/4
    Lmax /= 4;
end

# if numsimu == 1
#     skip_decrease = false;
# else
#     skip_decrease = true;
# end
skip_decrease = false;

## Calculating best grid search step size for SAGA_nice
function calculate_best_stepsize_SAGA_nice(prob, options ; skip, max_time, rep_number, batchsize, grid)
    old_skip = options.skip_error_calculation;
    old_tol = options.tol;
    old_max_iter = options.max_iter;
    old_max_epocs = options.max_epocs;
    old_max_time = options.max_time;
    old_rep_number = options.rep_number;
    old_batchsize = options.batchsize;

    options.repeat_stepsize_calculation = true;
    options.rep_number = rep_number;
    options.skip_error_calculation = skip;
    # options.tol = 10.0^(-16);
    options.max_iter = 10^8;
    options.max_time = max_time;
    options.batchsize = batchsize;
    SAGA_nice = initiate_SAGA_nice(prob, options);
    output = minimizeFunc_grid_stepsize(prob, SAGA_nice, options, grid=grid);

    options.repeat_stepsize_calculation = false;
    options.skip_error_calculation = old_skip;
    options.tol = old_tol;
    options.max_iter = old_max_iter;
    options.max_epocs = old_max_epocs;
    options.max_time = old_max_time;
    options.rep_number = old_rep_number;
    options.batchsize = old_batchsize;
    return output
end


### I) tau = 1 ###
#region
##---------- Computing step sizes ----------
# step_defazio = 1.0 / (3.0*(Lmax + n*mu));
# K = (4.0*Lmax) / (n*mu);
# step_hofmann = K / (2*Lmax*(1+K+sqrt(1+K^2)));
# # step_hofmann = 1.0 / (mu*n); # ridiculously large if mu is very small
# step_heuristic = 1.0 / (4.0*Lmax + n*mu);

# # Warning SAGA-nice too look for step size but method is called SAGA_nice
# step_gridsearch, = get_saved_stepsize(prob.name, "SAGA-nice", options)
# if step_gridsearch == 0.0
#     grid = [2.0^(25), 2.0^(23), 2.0^(21), 2.0^(19), 2.0^(17), 2.0^(15), 2.0^(13), 2.0^(11),
#             2.0^(9), 2.0^(7), 2.0^(5), 2.0^(3), 2.0^(1), 2.0^(-1), 2.0^(-3), 2.0^(-5),
#             2.0^(-7), 2.0^(-9), 2.0^(-11), 2.0^(-13), 2.0^(-15), 2.0^(-17), 2.0^(-19),
#             2.0^(-21), 2.0^(-23), 2.0^(-25), 2.0^(-27), 2.0^(-29), 2.0^(-31), 2.0^(-33)];
#     nbskip = closest_power_of_ten(round.(Int, n ));
#     output = calculate_best_stepsize_SAGA_nice(prob, options, skip=nbskip, max_time=60.0,
#                                                rep_number=5, batchsize=1, grid=grid);

#     step_gridsearch, = get_saved_stepsize(prob.name, "SAGA-nice", options);
# end

# stepsizes = [step_defazio, step_heuristic, step_gridsearch, step_hofmann];

# str_step_defazio = @sprintf "%.2e" step_defazio
# str_step_heuristic = @sprintf "%.2e" step_heuristic
# str_step_heuristic_gridsearch = @sprintf "%.2e" step_gridsearch
# str_step_hofmann = @sprintf "%.2e" step_hofmann
# method_names = [latexstring("\$b = 1 + \\gamma_\\mathrm{Defazio} \\ \\ \\: \\: = $str_step_defazio\$"),
#                 latexstring("\$b = 1 + \\gamma_\\mathrm{practical} \\ \\  = $str_step_heuristic\$"),
#                 latexstring("\$b = 1 + \\gamma_\\mathrm{grid search} = $str_step_heuristic_gridsearch\$"),
#                 latexstring("\$b = 1 + \\gamma_\\mathrm{Hofmann}   \\ \\, = $str_step_hofmann\$")];

# ##---------- SAGA_nice-1 runs ----------
# # options = set_options(tol=10.0^(-6), max_iter=10^8, max_epocs=10^8,
# #                       max_time=120.0,
# #                       skip_error_calculation=10^4,
# #                       batchsize=1,
# #                       regularizor_parameter = "normalized",
# #                       initial_point="zeros", # is fixed not to add more randomness
# #                       force_continue=true); # force continue if diverging or if tolerance reached

# # skip_error = calculate_skip_error.(stepsizes);
# # skip_error = [10^0, 10^1, 10^1, 10^1];
# # skip_error = [10^2, 10^2, 10^2, 10^2];
# # skip_error = [10^2, 10^3, 10^3, 10^3];
# # skip_error = [10^5, 10^5, 10^5, 10^5];
# skip_error = closest_power_of_ten.(round.(Int, n ./ ([1 1 1 1] ))) # 5 points per epoch
# # if numsimu == 1
#     # skip_error *= 10;
# # end
# # itercomplex = zeros(length(stepsizes), 1);
# itercomplex = zeros(length(stepsizes), numsimu);
# options.batchsize = 1;
# OUTPUTS = [];
# for idxstep in 1:length(stepsizes)
#     options.stepsize_multiplier = stepsizes[idxstep];
#     for idxsimu=1:numsimu
#         println("\n----- Simulation #", idxsimu, " -----");
#         options.skip_error_calculation = skip_error[idxstep]; # compute a skip error for each step size
#         SAGA_nice = initiate_SAGA_nice(prob, options); # separated implementation from SAGA
#         println("Current step size: ", method_names[idxstep], " = ", stepsizes[idxstep]);
#         output = minimizeFunc(prob, SAGA_nice, options, stop_at_tol=true, skip_decrease=skip_decrease);
#         println("---> Output fail = ", output.fail, "\n");
#         # itercomplex[idxstep] += output.iterations;
#         itercomplex[idxstep, idxsimu] = output.iterations;
#         output.name = string(method_names[idxstep]);
#         global OUTPUTS = [OUTPUTS; output];
#     end
# end
# avg_itercomplex = mean(itercomplex, dims=2);
# if numsimu > 1
#     std_itercomplex = std(itercomplex, dims=2);
#     ci_itercomplex = (1.96/sqrt(numsimu)) * std_itercomplex;
# else
#     ci_itercomplex = [0 0 0 0];
# end

# ## Saving the result of the simulations
# probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
# savename = string(probname, "-exp3_1-empcomplex-", numsimu, "-avg");
# save("$(default_path)$(savename).jld", "itercomplex", itercomplex, "ci_itercomplex", ci_itercomplex,
#      "OUTPUTS", OUTPUTS, "method_names", method_names,"stepsizes", stepsizes);

# ## Checking that all simulations reached tolerance
# fails = [OUTPUTS[i].fail for i=1:length(stepsizes)*numsimu];
# if all(s->(string(s)=="tol-reached"), fails)
#     println("Tolerance always reached")
# end

# ## Plotting one SAGA-nice simulation for each mini-batch size
# if numsimu == 1
#     # gr()
#     pyplot()
#     plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp3_1"); # Plot and save output
# end

# @printf "\n|  %s  | %s | %s |  %s   |\n" method_names[1] method_names[2] method_names[3] method_names[4]
# @printf "| %e  | %e  | %e  | %e |\n\n" stepsizes[1] stepsizes[2] stepsizes[3] stepsizes[4]
# @printf "| %d  | %d  | %d  | %d |\n\n" itercomplex[1] itercomplex[2] itercomplex[3] itercomplex[4]

# line1 =          "method name      |     b=1 + step_Defazio    | b=1 + step_heuristic |     b=1 + step_gridsearch    |    b=1 + step_Hofmann |\n";
# line3 = @sprintf "step size        |          %e        |        %e       |      %e     |     %e     |\n" stepsizes[1] stepsizes[2] stepsizes[3] stepsizes[4];
# line4 = @sprintf "total complexity |               %s             |           %s           |         %s           |         %s          |\n" format(itercomplex[1], commas=true) format(itercomplex[2], commas=true) format(itercomplex[3], commas=true) format(itercomplex[4], commas=true);
# line5 = @sprintf "CI delta         |               %d             |           %d           |         %d           |         %d          |\n" ci_itercomplex[1] ci_itercomplex[2] ci_itercomplex[3] ci_itercomplex[4]

# println("---------------------------------------------------------------------------------------------------------------------\n");
# println(line1);
# println("--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
# println(line3);
# println(line4);
# println(line5);
# println("number of simulations: $numsimu\n\n");

# open("./outputs/$probname-exp3_1.txt", "a") do file
#     write(file, "----------------------------------------------------------------------------------------------------------------------------\n");
#     write(file, line1);
#     write(file, "----------------------------------------------------------------------------------------------------------------------------\n");
#     write(file, line3);
#     write(file, line4);
#     write(file, line5);
#     write(file, "number of simulations: $numsimu\n\n");
#     write(file, "\n");
# end

#endregion

### II) tau = tau* ###
# Hofmann : tau = 20, gamma = gamma(20)
##---------- Computing step sizes ----------
tau_defazio = 1;
step_defazio = 1.0 / (3.0*(Lmax + n*mu));

tau_hofmann = 20;
K = (4.0*tau_hofmann*Lmax) / (n*mu);
step_hofmann = K / (2*Lmax*(1+K+sqrt(1+K^2)));
# step_hofmann = tau/(mu*n);

rho = ( n*(n - tau_hofmann) ) / ( tau_hofmann*(n-1) ); # Sketch residual
rightterm = (rho / n)*Lmax + ( (mu*n) / (4*tau_hofmann) ); # Right-hand side term in the max
heuristicbound = ( n*(tau_hofmann-1)*L + (n-tau_hofmann)*Lmax ) / ( tau_hofmann*(n-1) );
step_hofmann_heuristic = 0.25 / max(heuristicbound, rightterm);

## Is our optimal tau always one???
## YearPredictionMSD scaled + mu = 10^(-3) => 13
## YearPredictionMSD scaled + mu = 10^(-1) => 1245
tau_heuristic = round(Int, 1 + ( mu*(n-1) ) / ( 4*L ) );
# tau_heuristic = 20;
rho = ( n*(n - tau_heuristic) ) / ( tau_heuristic*(n-1) ); # Sketch residual
rightterm = (rho / n)*Lmax + ( (mu*n) / (4*tau_heuristic) ); # Right-hand side term in the max
heuristicbound = ( n*(tau_heuristic-1)*L + (n-tau_heuristic)*Lmax ) / ( tau_heuristic*(n-1) );
step_heuristic = 0.25 / max(heuristicbound, rightterm);

## Calculating best grid search step size for SAGA_nice with batchsize >= 1
# options = set_options(tol=10.0^(-6),
#                       max_iter=10^8,
#                       max_epocs=10^8,
#                       max_time=120.0,
#                       skip_error_calculation=10^5,
#                       regularizor_parameter = "normalized",
#                       initial_point="zeros",
#                       force_continue=true,
#                       batchsize=tau_heuristic);
options.batchsize = tau_heuristic;
if options.batchsize == 1
    method_name = "SAGA-nice";
elseif options.batchsize > 1
    method_name = string("SAGA-", options.batchsize, "-nice");
else
    error("Invalid batch size");
end

step_heuristic_gridsearch, = get_saved_stepsize(prob.name, method_name, options);
if step_heuristic_gridsearch == 0.0 || relaunch_gridsearch
    grid = [2.0^(25), 2.0^(23), 2.0^(21), 2.0^(19), 2.0^(17), 2.0^(15), 2.0^(13), 2.0^(11),
            2.0^(9), 2.0^(7), 2.0^(5), 2.0^(3), 2.0^(1), 2.0^(-1), 2.0^(-3), 2.0^(-5),
            2.0^(-7), 2.0^(-9), 2.0^(-11), 2.0^(-13), 2.0^(-15), 2.0^(-17), 2.0^(-19),
            2.0^(-21), 2.0^(-23), 2.0^(-25), 2.0^(-27), 2.0^(-29), 2.0^(-31), 2.0^(-33)];
    nbskip = closest_power_of_ten(round.(Int, n ./ tau_heuristic ));
    # nbskip = closest_power_of_ten(round.(Int, n ./ (skip_mult*tau_heuristic) ));
    output = calculate_best_stepsize_SAGA_nice(prob, options, skip=nbskip, max_time=60.0,
                                               rep_number=5, batchsize=tau_heuristic, grid=grid);
    step_heuristic_gridsearch, = get_saved_stepsize(prob.name, method_name, options);
end

# str_step_defazio = @sprintf "%.2e" step_defazio
# str_step_heuristic = @sprintf "%.2e" step_heuristic
# str_step_heuristic_gridsearch = @sprintf "%.2e" step_heuristic_gridsearch
# str_step_hofmann_heuristic = @sprintf "%.2e" step_hofmann_heuristic
# str_step_hofmann = @sprintf "%.2e" step_hofmann
# method_names = [latexstring("\$b_\\mathrm{Defazio} \\; \\; = 1 \\ \\ + \\gamma_\\mathrm{Defazio} \\ \\ \\: \\: = $str_step_defazio\$"),
#                 latexstring("\$b_\\mathrm{practical} \\, = $tau_heuristic + \\gamma_\\mathrm{practical} \\ \\ = $str_step_heuristic\$"),
#                 latexstring("\$b_\\mathrm{practical} \\, = $tau_heuristic + \\gamma_\\mathrm{grid search} = $str_step_heuristic_gridsearch\$"),
#                 latexstring("\$b_\\mathrm{Hofmann} = 20 + \\gamma_\\mathrm{practical} \\ \\ = $str_step_hofmann_heuristic\$"),
#                 latexstring("\$b_\\mathrm{Hofmann} = 20 + \\gamma_\\mathrm{Hofmann}  \\ \\, = $str_step_hofmann\$")];
# mini_batch_sizes = [tau_defazio, tau_heuristic, tau_heuristic, tau_hofmann, tau_hofmann];
# stepsizes = [step_defazio, step_heuristic, step_heuristic_gridsearch, step_hofmann_heuristic, step_hofmann];

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

##---------- SAGA_nice-1 runs ----------
# options = set_options(tol=10.0^(-6),
#                       max_time=120.0,
#                       skip_error_calculation=10^3,
#                       max_iter=10^8,
#                       max_epocs=10^8,
#                       regularizor_parameter="normalized", initial_point="zeros", force_continue=true);
# skip_error = [10^2, 10^2, 10^2, 10^2, 10^2];
# skip_error = [10^2, 10^3, 10^3, 10^3, 10^3];
# skip_error = [10^1, 10^4, 10^3, 10^3, 10^1]; # skip = n/(tau*10) approx 10 pass for 1 epoch

## Allowing user skip_error
if skip_error == [0 0 0 0]
    skip_error = closest_power_of_ten.(round.(Int, n ./ (5*mini_batch_sizes) )); # 5 points per epoch
end
println("---------------------------------- SKIP_ERROR ------------------------------------------");
println(skip_error);
println("----------------------------------------------------------------------------------------");

println("------------------------------- MINI-BATCH SIZES ---------------------------------------");
println(mini_batch_sizes);
println("----------------------------------------------------------------------------------------");

println("---------------------------------- STEP SIZES ------------------------------------------");
println(stepsizes);
println("----------------------------------------------------------------------------------------");

# skip_error = closest_power_of_ten.(round.(Int, n ./ (skip_mult*5*mini_batch_sizes))); # 5 points per epoch
# if numsimu == 1
    # skip_error *= 10;
    # skip_error = skip_error .* [10 10 10 1000];
# end
# itercomplex = zeros(length(stepsizes), 1);
itercomplex = zeros(length(stepsizes), numsimu);
OUTPUTS = [];
# options.force_continue = true;
for idxmethod in 1:length(stepsizes)
    options.stepsize_multiplier = stepsizes[idxmethod];
    for idxsimu=1:numsimu
        println("\n----- Simulation #", idxsimu, " -----");
        options.skip_error_calculation = skip_error[idxmethod]; # compute a skip error for each step size
        options.batchsize = mini_batch_sizes[idxmethod];
        SAGA_nice = initiate_SAGA_nice(prob, options);
        println("Current method: ", method_names[idxmethod], ", mini-batch size = ", mini_batch_sizes[idxmethod],
                ", step size = ", stepsizes[idxmethod]);
        output = minimizeFunc(prob, SAGA_nice, options, stop_at_tol=true, skip_decrease=skip_decrease);
        println("---> Output fail = ", output.fail, "\n");
        # itercomplex[idxmethod] += output.iterations;
        itercomplex[idxmethod, idxsimu] = output.iterations;
        # println("name = ", output.name)
        output.name = string(method_names[idxmethod]);
        global OUTPUTS = [OUTPUTS; output];
    end
end
avg_itercomplex = mean(itercomplex, dims=2);
avg_empcomplex = mini_batch_sizes .* avg_itercomplex;
if numsimu > 1
    empcomplex = mini_batch_sizes .* itercomplex;
    std_itercomplex = std(empcomplex, dims=2);
    ci_itercomplex = (1.96/sqrt(numsimu)) * std_itercomplex;
else
    empcomplex = avg_empcomplex;
    ci_itercomplex = [0 0 0 0 0];
end

## Saving the result of the simulations
probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
savename = string(probname, "-exp3_2-empcomplex-", numsimu, "-avg");
save("$(default_path)$(savename).jld", "itercomplex", itercomplex, "empcomplex", empcomplex, "ci_itercomplex", ci_itercomplex,
     "OUTPUTS", OUTPUTS, "method_names", method_names, "skip_error", skip_error,
     "stepsizes", stepsizes, "mini_batch_sizes", mini_batch_sizes);

## Checking that all simulations reached tolerance
fails = [OUTPUTS[i].fail for i=1:length(stepsizes)*numsimu];
if all(s->(string(s)=="tol-reached"), fails)
    println("Tolerance always reached")
end

## Plotting one SAGA-nice simulation for each mini-batch size
if numsimu == 1
    # gr()
    pyplot()
    plot_outputs_Plots(OUTPUTS, prob, options, suffix="-exp3.2"); # Plot and save output

    OUTPUTS_without_hofmann = OUTPUTS[1:3];
    # OUTPUTS_without_hofmann = [OUTPUTS_without_hofmann; OUTPUTS[3:5]];
    pyplot()
    plot_outputs_Plots(OUTPUTS_without_hofmann, prob, options, suffix="_without_hofmann-exp3.2"); # Plot and save output
end

# @printf "\nmethod name      | %s |     %s     |    %s    |  %s   |\n" method_names[1] method_names[2] method_names[3] method_names[4]
# @printf "mini-batch size  |            %d           |       %d      |      %d      |      %d      |\n\n" mini_batch_sizes[1] mini_batch_sizes[2] mini_batch_sizes[3] mini_batch_sizes[4]
# @printf "step size        |       %e      | %e | %e | %e |\n\n" stepsizes[1] stepsizes[2] stepsizes[3] stepsizes[4]
# @printf "total complexity |       %s        |   %s  |  %s  |  %s |\n\n" format(avg_empcomplex[1], commas=true) format(avg_empcomplex[2], commas=true) format(avg_empcomplex[3], commas=true) format(avg_empcomplex[4], commas=true)

# @printf "\nmethod name      | %s |     %s     |    %s    |  %s   |  %s   |\n" method_names[1] method_names[2] method_names[3] method_names[4] method_names[5]
# @printf "mini-batch size  |            %d           |       %d      |      %d      |      %d      |      %d      |\n" mini_batch_sizes[1] mini_batch_sizes[2] mini_batch_sizes[3] mini_batch_sizes[4] mini_batch_sizes[5]
# @printf "step size        |       %e      | %e | %e | %e | %e |\n" stepsizes[1] stepsizes[2] stepsizes[3] stepsizes[4] stepsizes[5]
# @printf "total complexity |       %s        |   %s  |  %s  |  %s |  %s |\n\n" format(avg_empcomplex[1], commas=true) format(avg_empcomplex[2], commas=true) format(avg_empcomplex[3], commas=true) format(avg_empcomplex[4], commas=true) format(avg_empcomplex[5], commas=true)

# line1 =          "method name      | b_Defazio + step_Defazio | b_heuristic + step_heuristic | b_heuristic + step_gridsearch | b_Hofmann + step_heuristic | b_Hofmann + step_Hofmann |\n"
# line2 = @sprintf "mini-batch size  |               %d               |             %d            |            %d            |             %d             |               %d              |\n" mini_batch_sizes[1] mini_batch_sizes[2] mini_batch_sizes[3] mini_batch_sizes[4] mini_batch_sizes[5]
# line3 = @sprintf "step size        |         %e          |       %e       |       %e       |        %e        |          %e        |\n" stepsizes[1] stepsizes[2] stepsizes[3] stepsizes[4] stepsizes[5]
# line4 = @sprintf "total complexity |             %s             |           %s            |         %s           |            %s             |             %s              |\n" format(avg_empcomplex[1], commas=true) format(avg_empcomplex[2], commas=true) format(avg_empcomplex[3], commas=true) format(avg_empcomplex[4], commas=true) format(avg_empcomplex[5], commas=true)
# line5 = @sprintf "CI delta         |               %d             |           %d           |         %d           |         %d          |         %d          |\n" ci_itercomplex[1] ci_itercomplex[2] ci_itercomplex[3] ci_itercomplex[4] ci_itercomplex[5]

line1 =          "method name      | b_Defazio + step_Defazio | b_heuristic + step_heuristic | b_heuristic + step_gridsearch | b_Hofmann + step_Hofmann |\n"
line2 = @sprintf "mini-batch size  |               %d               |             %d            |            %d            |               %d              |\n" mini_batch_sizes[1] mini_batch_sizes[2] mini_batch_sizes[3] mini_batch_sizes[4]
line3 = @sprintf "step size        |         %e          |       %e       |       %e       |        %e        |\n" stepsizes[1] stepsizes[2] stepsizes[3] stepsizes[4]
line4 = @sprintf "total complexity |             %s             |           %s            |         %s           |            %s             |\n" format(avg_empcomplex[1], commas=true) format(avg_empcomplex[2], commas=true) format(avg_empcomplex[3], commas=true) format(avg_empcomplex[4], commas=true)
line5 = @sprintf "CI delta         |               %d             |           %d           |         %d           |         %d          |\n" ci_itercomplex[1] ci_itercomplex[2] ci_itercomplex[3] ci_itercomplex[4]


println("--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
println(line1);
println("--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
println(line2);
println(line3);
println(line4);
println(line5);
println("number of simulations: $numsimu\n\n");

open("./outputs/$probname-exp3_2-complexity.txt", "a") do file
    write(file, "--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    write(file, line1);
    write(file, "--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    write(file, line2);
    write(file, line3);
    write(file, line4);
    write(file, line5);
    write(file, "number of simulations: $numsimu\n\n");
    write(file, "\n");
end