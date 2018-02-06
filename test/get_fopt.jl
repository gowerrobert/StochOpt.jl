using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")
# This is a script for finding and saving fsol, the minimum objective value, to each problem in problemnames.
# The parameters are chosen so that it performs 40 passes over the data.
## Basic parameters
maxiter=10^8;
max_time = 60*60*3; #60*60
max_epocs = 300; #50
printiters = true;
exacterror =false; repeat = false;
force_continue  = true ; # makes iterate continue even if surpassed previous fsol
tol = 10.0^(-16.0);
skip_error_calculation =0.0;   # number of iterations where error is not calculated (to save time!). Use 0 for default value
rep_number = 1;# number of times the optimization should be repeated.
options = MyOptions(tol,Inf,maxiter,skip_error_calculation,max_time,max_epocs,
printiters,exacterror,0,"normalized",0.0,false, force_continue,rep_number,0)
datapath = ""# "/local/rgower/libsvmdata/"
default_path = "./data/";
#problemnames =[ "mushrooms"] #,
problemnames =[  "a9a", "madelon", "phishing", "covtype", "gisette_scale", "w8a"] #,  liver-disorders_scale

for probname in problemnames
  prob =  load_logistic(probname,datapath,options);  # Loads logisitc
  options.batchsize =prob.numdata; ## load problem
  # prob.fsol = 0.0;
  method_name = "BFGS";
  output= minimizeFunc_grid_stepsize(prob, method_name, options, repeat);
  fsol = minimum(output.fs);#min(output.fs[end],fsol);
  savename = string(replace(prob.name, r"[\/]", "-"),"-fsol");
  save("$(default_path)$(savename).jld", "fsol", fsol)
end
