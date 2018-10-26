## Testing if our implementation of the gradient for the ridge regression problem is correct.
## See "load_ridge_regression.jl"
## Ref: https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/
using JLD
using Plots
using StatsBase
using Match
include("./src/StochOpt.jl") # WARNING: be careful of the path here
pgfplots()

## Basic parameters
options = set_options(tol=10.0^(-6.0), max_iter=10^8, max_time=300.0, max_epocs=30, rep_number=5);
options.batchsize = 1;

## Load problem
numdata = 100;
numfeatures = 60;
X = randn(numfeatures, numdata); # dxn input matrix (transpose of the design matrix) 
y = randn(numdata); # nx1 vector of labels
# x = rand(numfeatures);
# V = X.*((X'*x -y)');

probname = string("gauss-", numfeatures, "-", numdata);   # Data tested in paper: w8a mushrooms gisette_scale,  madelon  a9a  phishing  covtype splice  rcv1_train  liver-disorders_scale
prob = load_ridge_regression(X, y, probname, options, lambda=10.0, scaling="none");

## Testing gradient code
errorsacc = 0;
errorinplace = 0;
errorsacc_sym = 0;
errorinplace_sym = 0;
numtrials = 100;
grad = zeros(prob.numfeatures);
eps = 10.0^(-6);
for iteri = 1:numtrials
  s = sample(1:prob.numdata, options.batchsize, replace=false);
  x = rand(prob.numfeatures); # sample a anchor point of dimension d
  d = rand(prob.numfeatures); # sample a direction of dimension d
  gradd_est_finitediff = (prob.f_eval(x + eps*d, s) - prob.f_eval(x, s))/eps;
  gradd_est_finitediff_sym = (prob.f_eval(x + eps*d, s) - prob.f_eval(x - eps*d, s))/(2*eps);

  gradd = prob.g_eval(x, s)'*d;
  gradd2 = prob.g_eval!(x, s, grad)'*d;

  errorsacc += norm(gradd_est_finitediff - gradd);
  errorinplace += norm(gradd_est_finitediff - gradd2);

  errorsacc_sym += norm(gradd_est_finitediff_sym - gradd);
  errorinplace_sym += norm(gradd_est_finitediff_sym - gradd2);
end
errorsacc /= numtrials;
errorinplace /= numtrials;
errorsacc_sym /= numtrials;
errorinplace_sym /= numtrials;
println("--- One-side difference formula ---")
println("average grad error: ", errorsacc)
println("average grad inplace error: ", errorinplace)

println("--- Symmetric difference formula ---")
println("average grad error: ", errorsacc_sym)
println("average grad inplace error: ", errorinplace_sym)