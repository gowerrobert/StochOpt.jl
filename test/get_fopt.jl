using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")
# This is a script for finding and saving fsol, the minimum objective value, to each problem in problemnames.
# The parameters are chosen so that it performs 40 passes over the data.
## Basic parameters
# maxiter=10^8;
# max_time = 60*60*3; #60*60
# max_epocs = 500; #50
# printiters = true;
# exacterror =false; repeat = true;
# force_continue  = true ; # makes iterate continue even if surpassed previous fsol
# tol = 10.0^(-16.0);
# skip_error_calculation =20.0;   # number of iterations where error is not calculated (to save time!). Use 0 for default value
# rep_number = 1;# number of times the optimization should be repeated.
# options = MyOptions(tol,Inf,maxiter,skip_error_calculation,max_time,max_epocs,
# printiters,exacterror,0,"normalized",0.0,false, force_continue,rep_number,0)
options = set_options(tol =10.0^(-16.0), skip_error_calculation =20, exacterror =false, max_iter=10^8, max_time =60.0*60.0*3.0,
 max_epocs = 500, repeat_stepsize_calculation = true, rep_number =2);

datapath = ""# "/local/rgower/libsvmdata/"
default_path = "./data/";
#problemnames =[ "mushrooms"] #,
problemnames =[  "a9a"] #,  [  "a9a", "madelon", "phishing", "covtype", "gisette_scale", "w8a"]

 for probname in problemnames
  prob =  load_logistic(probname,datapath,options);  # Loads logisitc
  options.batchsize =prob.numdata; ## load problem
  # prob.fsol = 0.0;
  method_name = "BFGS";
  output= minimizeFunc_grid_stepsize(prob, method_name, options);
  options.embeddim =  [prob.numdata ,0.01];  #[0.9, 5];
  method_name = "BFGS_accel";
  output1= minimizeFunc_grid_stepsize(prob, method_name, options);
  OUTPUTS = [output ; output1];
  pgfplots()# gr() pyplot() # pgfplots() #plotly()
  plot_outputs_Plots(OUTPUTS,prob,options)
  fsol = minimum([output.fs output1.fs]);#min(output.fs[end],fsol);
  fsolfilename = get_fsol_filename(prob);
  save("$(fsolfilename).jld", "fsol", fsol)
 end
