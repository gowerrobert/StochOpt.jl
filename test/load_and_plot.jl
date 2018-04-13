## Load and plot previously saved data.
using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")

## Set default options and parameters
options = set_options();

## load problem
datapath = ""#
probnames = ["gisette_scale"]#[   "phishing", "madelon",  "a9a",  "mushrooms", "w8a", "gisette_scale"  ,"covtype", "rcv1_train"]# mushrooms
for probname in probnames
  prob =  load_logistic(probname,datapath,options);
  boot_method("-", prob,options);
  default_path = "./data/";   loadname= replace(prob.name, r"[\/]", "-");
  OUTPUTS = load("$(default_path)$(loadname).jld", "OUTPUTS");
##  Some code for editing which methods are plotted
     # OUTPUTNEW = OUTPUTS[1];
     # OUTPUTNEW = [OUTPUTNEW ; OUTPUTS[4:end]];
     # OUTPUTNEW = [OUTPUTNEW ; OUTPUTS[end]];
  pgfplots()# gr() pyplot() # pgfplots() #plotly()
  plot_outputs_Plots(OUTPUTS,prob,options)
 end






   # if(prob.numfeatures <5001.0)
   #   method_name = "SVRG2";
   #   output2= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
   #   OUTPUTS = [OUTPUTS ; output2];
   # end
   # default_path = "./data/";   savename= replace(prob.name, r"[\/]", "-");
   # save("$(default_path)$(savename).jld", "OUTPUTS",OUTPUTS);
     #
