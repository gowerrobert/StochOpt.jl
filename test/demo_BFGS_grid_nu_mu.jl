# A gridsearch that tries to identify the optimal mu and nu parameters.
# Since our theory does not apply to the BFGS deterministic update, we do not know a priori what are suitable
# bounds on the parameters mu and nu for the grid search.

using JLD
using Plots
using StatsBase
using Match
using LaTeXStrings
include("../src/StochOpt.jl")
## Basic parameters
maxiter=10^6;
max_time = 300;
max_epocs = 100;
printiters = true;
exacterror =true;
repeat = false;       # repeat the grid_search calculation for finding the stepsize
tol = 10.0^(-16.0);
skip_error_calculation =50.0;   # number of iterations where error is not calculated (to save time!). Use 0 for default value
rep_number = 2;# number of times the optimization should be repeated. This is because of julia just in time compiling
options = MyOptions(tol,Inf,maxiter,skip_error_calculation,max_time,max_epocs,
printiters,exacterror,0,"normalized",0.0,false, false,rep_number,0)
## load problem
datapath = ""#
# probname = "a9a";   # Data tested in paper: australian gisette_scale  w8a  madelon  a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders
#problems = ["w8a", "madelon", "covtype", "gisette_scale"];
problems = ["phishing"]
for probname in problems
    prob =  load_logistic(probname,datapath,options);  # Loads logisitc problem
    options.batchsize =prob.numdata;  # full batch
    ### Creating mus and nus grid
    # nus = nu_min: (nu_max-nu_min)/10 : nu_max;
    mu_max = 2.0;
    mu_min = 0.1;
    mus = round(mu_min: (mu_max-mu_min)/10 : mu_max,3);
    nu_max = 3;# prob.numdata;
    nu_min = 0.5;
    nus = round(nu_min: (nu_max-nu_min)/10 : nu_max,3);
    # lognus = log(nu_min): (log(nu_max)-log(nu_min))/10 : log(nu_max);
    # nus = round(exp(lognus),2);
    fendgrid = zeros(length(mus),length(nus));
    ### Getting the BFSG stepsize to use with all methods
    # beststep, savename = get_saved_stepsize(prob.name, "BFGS",options);
    # options.stepsize_multiplier =beststep;
    ###### Running accelerated methods
    for i = 1:length(mus)
        for j = 1:length(nus)
            options.embeddim =  [mus[i], nus[j]];
            method_name = "BFGS_accel";
            output1= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
            fendgrid[i,j] = output1.fs[end];
            OUTPUTS = [OUTPUTS ; output1];
        end
    end
    default_path = "./data/";
    indexm = findmin(fendgrid)
    (rowi , coli) =  ind2sub(fendgrid, indexm[2])
    mu_opt= mus[rowi];
    nu_opt= nus[coli];
    save("$(default_path)$(probname)_opt_mu_nu.jld", "fendgrid",fendgrid, "mu_opt",mu_opt, "nu_opt",nu_opt );
    # ###### Run opt
    fendgrid, mu_opt,nu_opt  = load("$(default_path)$(probname)_opt_mu_nu.jld", "fendgrid", "mu_opt", "nu_opt" );
    OUTPUTS = [];  # List of saved outputs
    options.skip_error_calculation =5;
    ######
    method_name = "BFGS";
    output1= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
    OUTPUTS = [OUTPUTS ; output1];

    options.embeddim =  [mu_opt, nu_opt];
    method_name = "BFGS_accel";
    output1= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
    OUTPUTS = [OUTPUTS ; output1];

    options.embeddim =  [0.9, 5];
    method_name = "BFGS_accel";
    output1= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
    OUTPUTS = [OUTPUTS ; output1];

    default_path = "./data/";   savename= replace(prob.name, r"[\/]", "-");
    pgfplots()
    plot_outputs_Plots(OUTPUTS,prob,options,max_epocs) # Plot and save output # max_epocs
    println("optimal (mu, nu) =  (", round(mu_opt,3), ", ", round(nu_opt,2), ")")
end
