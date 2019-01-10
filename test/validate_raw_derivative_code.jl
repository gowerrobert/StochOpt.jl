using JLD
using Plots
using StatsBase
using Match
include("../src/StochOpt.jl")
#
# probname = "australian"
# datapath = "./data/"
# name = string("lgstc_", probname);  # maybe add different regularizor later opts.regularizor
# X, y = loadDataset(datapath, probname);

numdata = 100;
numfeatures = 4;
X = rand(numfeatures, numdata);
y = convert.(Float64, rand(Bool, numdata));
miny = minimum(y);
y[findall(x->x==miny, y)]  .= -1;
y[findall(x->x > miny, y)] .= 1;
x = rand(numfeatures);

probname = string("gauss-", numfeatures, "-", numdata);
prob = load_logistic_from_matrices(X, y, probname, options, lambda=0.0, scaling="column-scaling");
sX = size(X);
numfeatures = sX[1];
numdata = sX[2];

# setup for tests
lambda = 0.001;

w = rand(numfeatures);
t =  logistic_phi(y.*X[:,:]'*w);
vec1 = (t.*(1 .-t));
expt = exp.(yXx);
vec2 = expt ./ ((1 .+ expt).^2);
H1 = (1/numdata)*X[:,:]*(vec1.*(X[:,:]'))  + lambda* Matrix{Float64}(I, numfeatures, numfeatures);
H2 = (1/numdata) .* X*(vec1.*(X'))  .+ lambda.* Matrix{Float64}(I, numfeatures, numfeatures);
H1-H2
## Full sparse Hessian
# w = [1 100 100]';
# H = rand(3,3);
g1 = rand(numfeatures);
g2 = rand(numfeatures);
d = rand(numfeatures);
eps = 10e-7;
g1 = logistic_grad!(X, y, w+eps*d, lambda, numdata, g1)
g2 = logistic_grad!(X, y, w, lambda, numdata, g2)
Hd=  (g1-g2)/eps
Hd2 = H2*d;

H3 = spzeros(numfeatures, numfeatures);
  # sparse(I, numfeatures, numfeatures);
logistic_hess!(X, y, w, lambda, numdata, g1, H3 )

Hd3 = H3*d;
display("Hd: ")
display(Hd)
display("Hd2: ")
display(Hd2)
display(norm(Hd-Hd2)/(norm(Hd)*numfeatures))
display(norm(Hd-Hd3)/(norm(Hd)*numfeatures))
# errorsacc = 0;
# numtrials = 100;
# H = spzeros(prob.numfeatures,prob.numfeatures)
# for iteri = 1:numtrials
#   s = sample(1:prob.numdata,5,replace=false);
#   w = rand(prob.numfeatures);
#   d = rand(prob.numfeatures);
#   g = rand(prob.numfeatures);
#   eps = 10.0^(-6);
#   Hdest= (prob.g_eval(w+eps.*d,s) - prob.g_eval(w,s))./eps;
#   Xx = X'*w;
#   yXx = y.*Xx;
#   t = logistic_phi(yXx) ;
#   H = X*(t.*(1 .-t).*(X')); #+sparse(I, length(w), length(w));
#   Hd = H*d;
#   global errorsacc+=norm(Hdest-Hd)./prob.numfeatures
# end
# errorsacc = errorsacc/numtrials;
# println("average Hess error: ", errorsacc)


## Hessian-vector for sparse matrices
# errorsacc = 0;
# numtrials = 100;
# for iteri = 1:numtrials
#   s = sample(1:prob.numdata,5,replace=false);
#   w = rand(prob.numfeatures);
#   d = rand(prob.numfeatures);
#   eps = 10.0^(-7);
#   g0 = X*(y.*(t .- 1));
#   gep = X*(y.*(logistic_phi(y.* (X'*(w+eps.*d))) .- 1));
#   Hdest= (gep-g0)./eps;
#   Hd = X*(t.*(1-t).*(X'*d));
#   errorsacc+=norm(Hdest-Hd)./prob.numfeatures
# end
# #errorsacc = errorsacc/numtrials;
# println("average Hess-vec error: ", errorsacc)


## Diagonal element alternatives

# Hv =zeros(numfeatures);
# D = zeros(numfeatures);
# errorsacc=0;
# HD= diag(X*(t.*(1-t).*X'));
# # for i =1:numfeatures # diagonal elemtn
# #    #D[i] =(Matrix(X[i,:])'*(t.*(1-t).*(X[i,:])))[1];
# #    D[i]=sum((X[i,:].^2).*(t.*(1-t)));
# # # works
# # # Hv[:]=  X*(t.*(1-t).*(X[i,:]));
# # # D[i] = Hv[i];
# # # doesnt' work
# #   #D= Matrix(X[i,:])'*(diagm(t.*(1-t))*(X[i,:]));
# #  #D= Matrix(X[i,:])'*((X[i,:]).*(t.*(1-t)));
# # # D=Matrix(X[i,:].^2)'*(t.*(1-t));
# # end
# D  =sum((X.^2)'.*(t.*(1-t)),1)';
# display(norm(HD -D))
