function boot_SPIN(prob::Prob,method, options::MyOptions, x::Array{Float64})
  method.stepsize = options.stepsize_multiplier;
  if(options.skip_error_calculation ==0.0)
    options.skip_error_calculation =10; # show 5 times per pass over the data
  end
  println("Skipping ", options.skip_error_calculation, " iterations per epoch")

  prob.g_eval!(x,1:prob.numdata,method.dn)
  return method
end



function initiate_SPIN(prob::Prob, options::MyOptions ; sketchsize= convert(Int64,floor(sqrt(prob.numfeatures))), sketchtype = "gauss")

  dn = zeros(prob.numfeatures);
  rhs = zeros(sketchsize);
  # prevx  = zeros(prob.numfeatures);
  S  = zeros(prob.numfeatures,sketchsize);
  HS  = zeros(prob.numfeatures,sketchsize);
  SHS  = zeros(sketchsize,sketchsize);
  grad= zeros(prob.numfeatures); # storing the previous gradient
  stepsize = 1.0;
  name = string("SPIN-",sketchsize, "-", sketchtype);#-",options.batchsize);
  if(sketchsize == prob.numfeatures)
      name = string("Newton");
  end
  return SPIN(1,1, name, descent_SPIN, boot_SPIN,  sketchsize, dn,  rhs, S,  HS, SHS, grad, stepsize,sketchtype)

end
