using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")
pgfplots()


## Basic parameters
options = set_options(tol=10.0^(-6.0), max_iter=10^8, max_time=300.0, max_epocs=30, rep_number=5);
options.batchsize = 5;
## load problem
# datapath = "./data";   # THIS RELATIVE PATH ONLY WORKS IF YOU ARE IN THE ROOT FOLDER !
datapath = ""# "/local/rgower/libsvmdata/"
# probname = "phishing";
# prob =  load_logistic(probname,datapath,options);  # Loads logisitc
numdata = 100;
numfeatures = 60;
X = rand(numfeatures, numdata);
y = rand(numdata);
x = rand(numfeatures);
V = X.*((X'*x -y)');

probname = string("gauss-", numfeatures, "-", numdata);   # Data tested in paper: w8a mushrooms gisette_scale,  madelon  a9a  phishing  covtype splice  rcv1_train  liver-disorders_scale
prob = load_ridge_regression(X, y, probname, options, lambda=10.0);
## Testing and benchmarking Jacobian code
Jac = zeros(prob.numfeatures, prob.numdata);#zeros(prob.numfeatures,prob.numfeatures);
Jac2 = zeros(prob.numfeatures, prob.numdata);
timeinplace = 0.0; time1 = 0.0; errorsacc = 0.0;
numtrials = 5;
for iteri = 1:numtrials
    x = rand(prob.numfeatures);
    s = sample(1:prob.numdata, options.batchsize, replace=false);
    tic();
    prob.Jac_eval!(x, s, Jac);
    time1 += toc();
    tic();
    for i in s
        Jac2[:,i] = prob.g_eval(x, [i]);
    end
    timeinplace += toc();
    # S = eye(prob.numfeatures)[:,C];
    # HS[:] = prob.Hess_opt(x,1:prob.numdata,S);
    errorsacc += norm(Jac-Jac2)/numtrials;
    Jac2[:] .= 0.0;
    Jac[:] .= 0.0;
end
println("timeinplace: ", timeinplace, "  time1: ", time1)
println("average Jacobian error: ", errorsacc)

## Benchmarking Hess_opt against Hess_opt!
# time1 = 0.0; time2 =0.0; errorsacc=0.0;
# numtrials = 10; batchsize= 1:prob.numdata; tau = 50;
# Hd = zeros(prob.numfeatures,tau);
# Hd1 = zeros(prob.numfeatures,tau);
# for iteri = 1:numtrials
#   s = 1:prob.numdata;#sample(1:prob.numdata,batchsize,replace=false);
#   w = rand(prob.numfeatures);
#   d = rand(prob.numfeatures,tau);
#   tic();
#   prob.Hess_opt!(w,s,d,Hd);
#   time2+= toc();
#   tic();
#   Hd1[:]=prob.Hess_opt(w,s,d);
#   time1+= toc();
#   errorsacc+= norm(Hd-Hd1)/numtrials;
# end
# #errorsacc = errorsacc/numtrials;
# println("Time1: ", time1, "  Timenew: ", time2)
# println("average error: ", errorsacc)

## Benchmarknig  Hess_C against Hess_C!
# println("average Hess-C subselection error: ")
# embeddim =20;
# HC = zeros(prob.numfeatures,embeddim);
# timeinplace = 0.0; time1 =0.0; errorsacc=0.0;
# numtrials = 10; batchsize= 1:prob.numdata;
# HS = zeros(prob.numfeatures,embeddim);
# for iteri = 1:numtrials
#   x = rand(prob.numfeatures);
#   C=  sample(1:prob.numfeatures,embeddim,replace=false);#
#   tic();
#   HS[:] = prob.Hess_C(x,1:prob.numdata,C);
#   time1 += toc();
#   tic();
#   prob.Hess_C!(x,1:prob.numdata,C,HC);
#   timeinplace+= toc();
#   # S = eye(prob.numfeatures)[:,C];
#   # HS[:] = prob.Hess_opt(x,1:prob.numdata,S);
#   errorsacc+=norm(HS-HC)/numtrials;
# end
# println("timeinplace: ", timeinplace, "  time1: ", time1)
# println("average error: ", errorsacc)

## Benchmarking Hess_D against Hess_D!
#
# println("average Hess_eval error: ")
# embeddim =20;
# H2 = zeros(prob.numfeatures);#zeros(prob.numfeatures,prob.numfeatures);
# H1 = zeros(prob.numfeatures);
# timeinplace = 0.0; time1 =0.0; errorsacc=0.0;
# numtrials = 100; batchsize= 500;
# for iteri = 1:numtrials
#   x = rand(prob.numfeatures);
#   tic();
#   H1[:] = prob.Hess_D(x,1:batchsize);
#   time1 += toc();
#   tic();
#   prob.Hess_D!(x,1:batchsize,H2);
#   timeinplace+= toc();
#   # S = eye(prob.numfeatures)[:,C];
#   # HS[:] = prob.Hess_opt(x,1:prob.numdata,S);
#   errorsacc+=norm(H1-H2)/numtrials;
#   #H2[:] .=0.0;
# end
# println("timeinplace: ", timeinplace, "  time1: ", time1)
# println("average error: ", errorsacc)

# Benchmarknig  Hess_eval against Hess_eval!
# println("average Hess_eval error: ")
# embeddim =20;
# H2 = spzeros(prob.numfeatures, prob.numfeatures);#zeros(prob.numfeatures,prob.numfeatures);
# H1 = zeros(prob.numfeatures,prob.numfeatures);
# timeinplace = 0.0; time1 =0.0; errorsacc=0.0;
# numtrials = 10; batchsize= 50;
# for iteri = 1:numtrials
#   x = rand(prob.numfeatures);
#   tic();
#   H1[:] = prob.Hess_eval(x,1:batchsize);
#   time1 += toc();
#   tic();
#   prob.Hess_eval!(x,1:batchsize,H2);
#   timeinplace+= toc();
#   # S = eye(prob.numfeatures)[:,C];
#   # HS[:] = prob.Hess_opt(x,1:prob.numdata,S);
#   errorsacc+=norm(H1-H2)/numtrials;
#   #H2[:] .=0.0;
# end
# println("timeinplace: ", timeinplace, "  time1: ", time1)
# println("average error: ", errorsacc)
#timeinplace: 31.985306602999998  time1: 40.01966824
# ## Plotting and testing the objective function
# s = sample(1:prob.numdata,5,replace=false);
# numsteps = 100;
# x = zeros(prob.numfeatures);
# d = rand(prob.numfeatures); d= d/norm(d);
# t = -2:0.1:2;
# fs= map(a -> prob.f_eval(x.+a*d,s), t);
# plot(t, fs)
#
# Testing gradient code
# errorsacc = 0;
# errorinplace =0;
# numtrials = 20;
# grad = zeros(prob.numfeatures);
# for iteri = 1:numtrials
#   s = sample(1:prob.numdata,5,replace=false);
#   x = rand(prob.numfeatures);
#   d = rand(prob.numfeatures);
#   eps = 10.0^(-6);
#   graestdd= (prob.f_eval(x+eps.*d,s) - prob.f_eval(x,s))/eps;
#   gradd = prob.g_eval(x,s)'*d;
#   gradd2 = prob.g_eval!(x,s,grad)'*d;
#   errorsacc = errorsacc + norm(graestdd-gradd);
#   errorinplace = errorinplace + norm(gradd2-gradd);
# end
# errorsacc = errorsacc/numtrials;
# errorinplace = errorinplace/numtrials;
# println("average grad error: ", errorsacc)
# println("average grad inplace error: ", errorinplace)
#
#
##Testing full Hessian code
# errorsacc = 0;
# numtrials = 100;
# H = zeros(prob.numfeatures,prob.numfeatures)
# for iteri = 1:numtrials
#   s = sample(1:prob.numdata,5,replace=false);
#   w = rand(prob.numfeatures);
#   d = rand(prob.numfeatures);
#   eps = 10.0^(-6);
#   Hdest= (prob.g_eval(w+eps.*d,s) - prob.g_eval(w,s))./eps;
#   Hd = prob.Hess_eval!(w,s,H)*d;
#   errorsacc+=norm(Hdest-Hd)./prob.numfeatures
# end
# errorsacc = errorsacc/numtrials;
# println("average Hess error: ", errorsacc)
# #
# # #Testing Hessian-vector product code
# errorsacc = 0;
# numtrials = 100; batchsize= 10;
# Hd = zeros(prob.numfeatures);
# Hd1 = zeros(prob.numfeatures);
# Hdest= zeros(prob.numfeatures);
# for iteri = 1:numtrials
#   s = sample(1:prob.numdata,batchsize,replace=false);
#   w = rand(prob.numfeatures);
#   d = rand(prob.numfeatures);
#   Hd[:] = zeros(prob.numfeatures);
#   eps = 10.0^(-7);
#   Hdest[:]= (prob.g_eval(w+eps.*d,s) - prob.g_eval(w,s))./eps;
# #  Hd1[:]=prob.Hess_opt(w,s,d);
#   prob.Hess_opt!(w,s,d,Hd);
#   errorsacc+=norm(Hdest-Hd)./prob.numfeatures
# end
# #errorsacc = errorsacc/numtrials;
# println("average Hess-vec error: ", errorsacc)

#
## Testing Hessian-opt against Hessian diagonal product
# numtrials = 10;
# w0 = rand(prob.numfeatures);
# Hdd = zeros(prob.numfeatures);
# Hddest= zeros(prob.numfeatures);
# d = zeros(prob.numfeatures);
# errorsacc = 0;
# for iteri = 1:numtrials
#   s = sample(1:prob.numdata,10,replace=false);
#   w0[:] = rand(prob.numfeatures);
#   Hdd[:] .= 0.0;
#   d[:] .= 0.0;
#   prob.Hess_D!(w0,s,Hddest);
#   #Hddest=diag(prob.Hess_eval(w0,s));
#   for j = 1:prob.numfeatures
#     d[j] = 1;
#     Hdd[j] = (d'*prob.Hess_opt(w0,s,d))[1];
#     d[j]=0;
#   end
#   # display(Hdd')
#   # display(Hddest')
#    errorsacc+=norm(Hddest-Hdd)./prob.numfeatures;
# end
# errorsacc = errorsacc/numtrials;
# println("average Hess-opt - Hess_D error: ", errorsacc)

# ## Testing Hessian-vector-vector against diagonal product
# errorsacc = 0;
# numtrials = 10;
#
# for iteri = 1:numtrials
#   s = sample(1:prob.numdata,5,replace=false);
#   w0 = rand(prob.numfeatures);
#   Hdd = zeros(prob.numfeatures);
#   d = zeros(prob.numfeatures);
#     Hddest= prob.Hess_D(w0,s);
#   for j = 1:prob.numfeatures
#     d[j] = 1;
#     Hdd[j] = prob.Hess_vv(w0,s,d)[1];
#     d[j]=0;
#   end
#   errorsacc+=norm(Hddest-Hdd)./prob.numfeatures
# end
# #errorsacc = errorsacc/numtrials;
# println("average Hess-vec-vec error: ", errorsacc)

#
# ## Testing Hessian-vector-vector against Hess_opt
# errorsacc = 0;
# numtrials = 10;
#
# for iteri = 1:numtrials
#   s = sample(1:prob.numdata,5,replace=false);
#   w0 = rand(prob.numfeatures);
#   Hdd = zeros(prob.numfeatures);
#   d = rand(prob.numfeatures,2);
#   dHdfull =d'*prob.Hess_eval(w0,s)*d;
#   dHd= d'*prob.Hess_opt(w0,s,d);
#   dHdest = prob.Hess_vv(w0,s,d);
#   errorest=norm(dHd-dHdest)./prob.numfeatures
# end
# #errorsacc = errorsacc/numtrials;
