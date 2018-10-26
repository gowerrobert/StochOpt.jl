## Testing if our implementation of the gradient for the ridge regression problem is correct.
## See "load_ridge_regression.jl"
## Ref: https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/
using JLD
using Plots
using StatsBase
using Match
include("./src/StochOpt.jl") # WARNING: be careful of the path here
pgfplots()

### Basic parameters
options = set_options(tol=10.0^(-6.0), max_iter=10^8, max_time=300.0, max_epocs=30, rep_number=5);
options.batchsize = 3;

### Load problem
numdata = 100;
numfeatures = 60;
X = randn(numfeatures, numdata); # dxn input matrix (transpose of the design matrix) 
y = randn(numdata); # nx1 vector of labels
# x = rand(numfeatures);
# V = X.*((X'*x -y)');

probname = string("gauss-", numfeatures, "-", numdata);   # Data tested in paper: w8a mushrooms gisette_scale,  madelon  a9a  phishing  covtype splice  rcv1_train  liver-disorders_scale
prob = load_ridge_regression(X, y, probname, options, lambda=10.0, scaling="none");

### Testing gradient code
#region
diffformula = 2; # 1 for one-sided, 2 for symmetric
eps = 10.0^(-6);
numtrials = 1000;

errorsacc = 0.0;        # absolute error
errorinplace = 0.0;     # absolute error for inplace
relerrorsacc = 0.0;     # relative error
relerrorinplace = 0.0;  # relative error for inplace
grad = zeros(prob.numfeatures);
for iteri = 1:numtrials
  s = sample(1:prob.numdata, options.batchsize, replace=false);
  x = rand(prob.numfeatures); # sample a anchor point of dimension d
  d = rand(prob.numfeatures); # sample a direction of dimension d

  ## Our implementations of the gradient
  gradd = prob.g_eval(x, s)'*d;
  gradd2 = prob.g_eval!(x, s, grad)'*d; # inplace impementation

  ## Finite difference estimate of the gradient
  if diffformula == 1
    gradd_est_finitediff = (prob.f_eval(x + eps*d, s) - prob.f_eval(x, s))/eps;
  elseif diffformula == 2
    gradd_est_finitediff = (prob.f_eval(x + eps*d, s) - prob.f_eval(x - eps*d, s))/(2*eps);
  else
    error("Chose a valide finite difference formula: 1 or 2");
  end
 
  ## Error incrementation
  errorsacc += norm(gradd_est_finitediff - gradd);
  errorinplace += norm(gradd_est_finitediff - gradd2);
  relerrorsacc += norm(gradd_est_finitediff - gradd)/norm(gradd_est_finitediff); # Computing relative error might be better because we might have a function taking tiny values (TO BE CHECKED?)
  relerrorinplace += norm(gradd_est_finitediff - gradd2)/norm(gradd_est_finitediff);
end
errorsacc /= numtrials;
errorinplace /= numtrials;
relerrorsacc /= numtrials;
relerrorinplace /= numtrials;

if diffformula == 1
  println("--- One-side difference formula ---");
elseif diffformula == 2
  println("--- Symmetric difference formula ---");
end
println("average grad absolute error: ", errorsacc);
println("average grad inplace absolute error: ", errorinplace);
println("average grad relative error: ", relerrorsacc);
println("average grad inplace relative error: ", relerrorinplace);

## Remark:
## - the one_sided difference error should be of size eps
## - the symmetric difference error should be of size eps^2
#endregion



### Testing and benchmarking Jacobian codes
#region
numtrials = 100;

Jac = zeros(prob.numfeatures, prob.numdata); # inplace estimate
Jac_est = zeros(prob.numfeatures, prob.numdata); # g_eval has already been tested
timeinplace = 0.0;
time1 = 0.0;
errorsacc = 0.0;
relerrorsacc = 0.0;
for iteri = 1:numtrials
    x = rand(prob.numfeatures);
    s = sample(1:prob.numdata, options.batchsize, replace=false);
    
    ## Our implementations of the Jacobian
    tic();
    prob.Jac_eval!(x, s, Jac);
    time1 += toc();
    tic();

    ## Jacobian estimate by concatenation of individual gradients
    for i in s
      Jac_est[:, i] = prob.g_eval(x, [i]);
    end
    timeinplace += toc();

    ## Error incrementation
    errorsacc += norm(Jac_est - Jac);
    relerrorsacc += norm(Jac_est - Jac)/norm(Jac_est);
    Jac_est[:] .= 0.0;
    Jac[:] .= 0.0;
end
errorsacc /= numtrials;
relerrorsacc /= numtrials;
println("timeinplace: ", timeinplace, "  time1: ", time1)
println("average Jacobian absolute error: ", errorsacc)
println("average Jacobian relative error: ", relerrorsacc)
#endregion

