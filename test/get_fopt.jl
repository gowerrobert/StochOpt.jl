using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")
# This is a script for finding and saving fsol, the minimum objective value, to each problem in problemnames.
# The parameters are chosen so that it performs 40 passes over the data.
## Basic parameters
maxiter=10^8;
max_time = 60*60*60*3; #60*60
max_epocs = 50; #50
printiters = true;
exacterror =true; repeat = false;
tol = 10.0^(-8.0);
skip_error_calculation =0.0;   # number of iterations where error is not calculated (to save time!). Use 0 for default value
rep_number = 1;# number of times the optimization should be repeated.
options = MyOptions(tol,Inf,maxiter,skip_error_calculation,max_time,max_epocs,
printiters,exacterror,0,"normalized",0.0,false, false,rep_number,0)
options.batchsize =100;
options.embeddim = 10; ## load problem
datapath = ""# "/local/rgower/libsvmdata/"
default_path = "./data/";
problemnames =[ "heart_scale" ];# "a9a", "mushrooms", "phishing", "covtype",  done already "gisette_scale", "rcv1_train", "", "a9a", liver-disorders_scale
repeat =true;
fsol = 0.0;
for probname in problemnames
  prob =  load_logistic(probname,datapath,options);  # Loads logisitc
  method_name = "SVRG";
  output= minimizeFunc_grid_stepsize(prob, method_name, options, repeat);
  fsol = output.fs[end];#min(output.fs[end],fsol);
  savename = string(replace(prob.name, r"[\/]", "-"),"-fsol");
  save("$(default_path)$(savename).jld", "fsol", fsol)
end
