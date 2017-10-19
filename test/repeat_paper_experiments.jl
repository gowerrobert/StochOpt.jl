using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")
# This script repeats the experiments from the paper
#          Tracking the gradients using the Hessian: A new look at variance reducing stochastic methods
# WARNING: on 24 core machine, this takes 2 days to run!
# WARNING 2: You will have to download the data of covtype, gisette_scale, and rcv1_train before running this code. See README
maxiter=10^8;
max_time = 350;
max_epocs = 20;
printiters = true;
exacterror =true; repeat = false;
tol = 10.0^(-6.0);
skip_error_calculation =0.0;   # number of iterations where error is not calculated (to save time!)
rep_number =10;
precondition = false;
options = MyOptions(tol,Inf,maxiter,skip_error_calculation,max_time,max_epocs,
printiters,exacterror,0,"normalized",0.0,precondition, false,rep_number,0.0)
options.batchsize =100;  #"onep";
options.embeddim =10; #percentage of featurenum
datapath = ""#
probnames =[  "a9a", "phishing"  , "mushrooms","splice","w8a","madelon", "gisette_scale", "covtype", "rcv1_train"];
for probname in probnames
  prob =  load_logistic(probname,datapath,options);  # Loads logisitc
  ## Running methods
  OUTPUTS = [];
  method_name = "SVRG";
  output= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
  OUTPUTS = [OUTPUTS ; output];
  # # #
  method_name = "2D";
  output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
  OUTPUTS = [OUTPUTS ; output3];
  # # #
  method_name = "2Dsec";
  output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
  OUTPUTS = [OUTPUTS ; output3];
  # # # # #
  method_name = "CMgauss";
  output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
  OUTPUTS = [OUTPUTS ; output3];
  # #
  method_name = "CMprev";
  output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
  OUTPUTS = [OUTPUTS ; output3];
  # # # #
  method_name = "DFPgauss";
  output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
  OUTPUTS = [OUTPUTS ; output3];
  # # ##
  method_name = "DFPprev";
  output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
  OUTPUTS = [OUTPUTS ; output3];
  # #
  if(prob.numfeatures <5001.0)
    method_name = "SVRG2";
    output2= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
    OUTPUTS = [OUTPUTS ; output2];
  end
  default_path = "./data/";   savename= replace(prob.name, r"[\/]", "-");
  save("$(default_path)$(savename).jld", "OUTPUTS",OUTPUTS);
  savename = string(replace(prob.name, r"[\/]", "-")) ;
  pgfplots()
#  gr()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS,prob,options,20)

end
