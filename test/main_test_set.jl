using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")
## Basic parameters
maxiter=10^8;
max_time = 350;
max_epocs = 20;
printiters = true;
exacterror =false; repeat = false;
tol = 10.0^(-6.0);
skip_error_calculation =0.0;   precondition = false;# number of iterations where error is not calculated (to save time!)
options = MyOptions(tol,Inf,maxiter,skip_error_calculation,max_time,max_epocs,printiters,exacterror,0,"normalized",0.0,precondition)
options.batchsize =10;
## load problem
datapath = ""# "/local/rgower/libsvmdata/"
probname = "madelon";# gisette_scale     madelon
# a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders_scale
prob =  load_logistic(probname,datapath,options);  # Loads logisitc
# load test set
testprob =  load_logistic(string(probname,".t"),datapath,options);  # Loads logisitc
## Running methods
OUTPUTS = [];
methods = [ "SVRG", "SVRG2", "SVRG2D", "SVRG2sec", "embed"]
w = zeros(prob.numfeatures)
te = test_error(testprob,w);
println("test error: ", te)
for method_name in methods
  savename = string(replace(prob.name, r"[\/]", "-"),'-',method_name,"-stepsize") ;
  options.stepsize_multiplier = get_saved_stepsize(prob.name, method_name);
  output= minimizeFunc_with_test(prob, testprob, method_name, options);
  OUTPUTS = [OUTPUTS ; output];
end
# #
default_path = "./data/";   savename= string(replace(prob.name, r"[\/]", "-"),"test");
save("$(default_path)$(savename).jld", "OUTPUTS",OUTPUTS);
# #PyPlot PGFPlots Plotly GR
gr()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS,testprob,10)
