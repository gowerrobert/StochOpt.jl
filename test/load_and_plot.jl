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
<<<<<<< HEAD
probnames = ["gisette_scale"]#[   "phishing", "madelon",  "a9a",  "mushrooms", "w8a", "gisette_scale"  ,"covtype", "rcv1_train"]# mushrooms
for probname in probnames
  prob =  load_logistic(probname,datapath,options);
  boot_method("-", prob,options);
  default_path = "./data/";   loadname= replace(prob.name, r"[\/]", "-");
=======
probnames = ["phishing", "madelon",  "a9a",  "mushrooms", "phishing", "w8a", "gisette_scale"  ,"covtype"]#  rcv1_train  liver-disorders_scale
# for probname in probnames
  probname = "madelon";
  name = string("lgstc_",  probname);
  # prob =  load_logistic(probname,datapath,options);
  # boot_method("-", prob,options);
  default_path = "./data/";   loadname= replace(name, r"[\/]", "-");
>>>>>>> 93946ba328b33a911610eab7593ce76e3996c39a
  OUTPUTS = load("$(default_path)$(loadname).jld", "OUTPUTS");
##  Some code for editing which methods are plotted
     # OUTPUTNEW = OUTPUTS[1];
     # OUTPUTNEW = [OUTPUTNEW ; OUTPUTS[4:end]];
     # OUTPUTNEW = [OUTPUTNEW ; OUTPUTS[end]];
  pgfplots()# gr() pyplot() # pgfplots() #plotly()
<<<<<<< HEAD
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
=======
  plot_outputs_Plots(OUTPUTS,prob,options,20)
 # end
>>>>>>> 93946ba328b33a911610eab7593ce76e3996c39a
