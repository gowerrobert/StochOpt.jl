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
exacterror =true; repeat = false;
tol = 10.0^(-6.0);
skip_error_calculation =0.0;   # number of iterations where error is not calculated (to save time!)
precondition = false; # Using a precondition/quasi-Newton type
rep_number = 10;# number of times the optimization should be repeated. Only the average is reported.
options = MyOptions(tol,Inf,maxiter,skip_error_calculation,max_time,max_epocs,
printiters,exacterror,0,"normalized",0.0,precondition, false,rep_number,0)
options.batchsize =100;
options.embeddim = 10; # The dimension of the S embedding matrix
## load problem
datapath = ""# "/local/rgower/libsvmdata/"
probname = "w8a";# gisette_scale   w8a  madelon splice
# a9a  phishing  covtype mushrooms  rcv1_train  liver-disorders_scale
prob =  load_logistic(probname,datapath,options);  # Loads logisitc
## Running methods
OUTPUTS = [];
# #
method_name = "SVRG";
output= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output];

# # # # #
method_name = "2D";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
# # #
method_name = "2Dsec";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
##
method_name = "CMcoord";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
# # # # #
method_name = "CMgauss";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
# # #
method_name = "CMprev";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
# # # #
method_name = "DFPcoord";
output3= minimizeFunc_grid_stepsize(prob, method_name, options,repeat);
OUTPUTS = [OUTPUTS ; output3];
#####
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
pgfplots()# gr() pyplot() # pgfplots() #plotly()
plot_outputs_Plots(OUTPUTS,prob,options,20)
