name = string("lgstc_", probname);  # maybe add different regularizor later opts.regularizor
datafile = string(datapath, probname); println("loading:  ", datafile)
X, y = loadDataset(datafile);
stdX = std(X, 2);
#replace 0 in std by 1 incase there is a constant feature
ind = (0.==stdX); stdX[ind] = 1.0; # Testing for a zero std
X[:, :]= (X.-mean(X, 2))./stdX; # Centering and scaling the data.
X = [X; ones(size(X, 2))'];
sX = size(X);
numfeatures = sX[1];
numdata = sX[2];

# setup for tests
w = rand(numfeatures);
Xx = X'*w;
yXx = y.*Xx;
t = logistic_phi(yXx) ;


## Full sparse Hessian
H = X*(t.*(1-t).*(X'))+speye(length(w));

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
