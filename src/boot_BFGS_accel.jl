function boot_BFGS(prob::Prob,method::Method, options::MyOptions)

  method.diffpnt = zeros(prob.numfeatures);
  method.prevx  = zeros(prob.numfeatures);
  method.gradsamp= zeros(prob.numfeatures); # storing the previous gradient
#  method.prevx  = zeros(prob.numfeatures,embeddim+1);# 1st position contain previous outer iterate, the 2:embedded contain the previous embedding matrix
  method.aux = zeros(prob.numfeatures,prob.numfeatures);  # stores the V_k's
  method.HSi = zeros(prob.numfeatures,prob.numfeatures);  # stores the Y_k's
  method.H = eye(prob.numfeatures);  # Store inverse Hessian approximation
  method.S = zeros(prob.numfeatures);  # Stores the difference between gradients
  method.HS= zeros(prob.numfeatures);  # Stores product of H*dy difference between gradients
  method.name = string("BFGS");#-",options.batchsize);
  method.gradsperiter = 3*prob.numfeatures;
  
#  method.gradsperiter = (embeddim+2)*options.batchsize+(embeddim+2)*prob.numdata/method.numinneriters+1; #includes the cost of performing the Hessian vector product.
  method.stepmethod = descent_BFGS;
  return method;
end
