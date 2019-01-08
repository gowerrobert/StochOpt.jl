using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")
# This script repeats the experiments from the paper
#          Tracking the gradients using the Hessian: A new look at variance reducing stochastic methods
# WARNING: on a 16 core machine, this takes 2 days to run!
# WARNING 2: You will have to download the data of covtype, gisette_scale, and rcv1_train before running this code. See README

# Parameter and options
options = set_options(max_iter=10^8, max_time=350.0, max_epocs=30, repeat_stepsize_calculation=false, rep_number=10);
options.batchsize = 100;
options.embeddim = 10; #percentage of featurenum
datapath = "./data/"#
probnames = ["phishing", "a9a", "mushrooms", "splice", "w8a", "madelon", "gisette_scale", "covtype", "rcv1_train"];
for probname in probnames
    options.batchsize = 100;
    options.skip_error_calculation = 0;   # number of iterations where error is not calculated (to save time!)
    prob = load_logistic(datapath, probname, options);  # Loads logisitc
    ## Running methods
    OUTPUTS = [];
    method_names = ["SVRG", "2D", "2Dsec", "CMprev", "CMgauss", "AMgauss", "AMprev"];  # Curvature matching methods: CMgauss,  CMprev
    for method_name in method_names
        output2 = minimizeFunc_grid_stepsize(prob, method_name, options)
        OUTPUTS = [OUTPUTS; output2];
    end
    if(prob.numfeatures < 5001.0)
        output2 = minimizeFunc_grid_stepsize(prob, "SVRG2", options);
        OUTPUTS = [OUTPUTS; output2];
    end
    options.batchsize = prob.numdata;
    options.skip_error_calculation = 1;
    method_name = "grad";
    output = minimizeFunc_grid_stepsize(prob, method_name, options);
    OUTPUTS = [OUTPUTS; output];
    default_path = "./data/";
    savename = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
    save("$(default_path)$(savename).jld", "OUTPUTS", OUTPUTS);
    gr()
#  gr()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS, prob, options, datapassbnd = 20)
end
