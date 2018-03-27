## Load and plot previously saved data.
using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")
maxiter=10^8;
max_time = 350;
max_epocs = 20;
printiters = true;
exacterror =true; repeat = false;
tol = 10.0^(-6.0);
skip_error_calculation =0.0;   # number of iterations where error is not calculated (to save time!)
precondition = false; # Using a precondition/quasi-Newton type
rep_number = 10;   # number of iterations where error is not calculated (to save time!)
options = MyOptions(tol,Inf,maxiter,skip_error_calculation,max_time,max_epocs,
printiters,exacterror,0,"normalized",0.0,precondition, false,rep_number,0)
options.batchsize =100;
## load problem
datapath = ""#
probnames = ["phishing", "madelon",  "a9a",  "mushrooms", "phishing", "w8a", "gisette_scale"  ,"covtype"]#  rcv1_train  liver-disorders_scale
# for probname in probnames
  probname = "madelon";
  name = string("lgstc_",  probname);
  # prob =  load_logistic(probname,datapath,options);
  # boot_method("-", prob,options);
  default_path = "./data/";   loadname= replace(name, r"[\/]", "-");
  OUTPUTS = load("$(default_path)$(loadname).jld", "OUTPUTS");
  pgfplots()# gr() pyplot() # pgfplots() #plotly()
  plot_outputs_Plots(OUTPUTS,prob,options,20)
 # end
