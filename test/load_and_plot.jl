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
datapath = ""# "/local/rgower/libsvmdata/"
#probnames = ["rcv1_train" ]
probnames = ["phishing", "madelon",  "a9a",  "mushrooms", "phishing", "w8a", "gisette_scale"  ,"covtype"]#  rcv1_train  liver-disorders_scale
for probname in probnames
  prob =  load_logistic(probname,datapath,options);
  boot_method("-", prob,options);
  default_path = "./data/";   loadname= replace(prob.name, r"[\/]", "-");
  OUTPUTS = load("$(default_path)$(loadname).jld", "OUTPUTS");
  #Swap order of SVRG2 to last place to maintain consistent markers in plot_outputs_Plots
  # OUTPUTS2 = [];
  # OUTPUTS2 = [OUTPUTS[1] ; OUTPUTS[3] ; OUTPUTS[4] ;OUTPUTS[5] ; OUTPUTS[2]];
  # OUTPUTS2[2].name = "SVRGDiag"; OUTPUTS2[3].name = "SVRGSec"; OUTPUTS2[4].name = "SVRGEmb";
  # OUTPUTS2[end].name = "SVRGHess";
  pgfplots()# gr() pyplot() # pgfplots() #plotly()
  plot_outputs_Plots(OUTPUTS,prob,options,20)
 end
