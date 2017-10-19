function descent_2D(x::Array{Float64},prob::Prob,options::MyOptions,method::Method, N::Int64,d::Array{Float64})
  # SVRG2D: A second order variant that uses a diagonal Hessian matrix
if(N%method.numinneriters ==1)# Reset reference point, grad estimate and Hessian estimate
    method.prevx[:]  = x;
    method.grad[:] = prob.g_eval(x,1:prob.numdata);
    method.H[:] =  prob.Hess_D(x, 1:prob.numdata);
    d[:]= -method.grad;
  else     # SVRG2D inner step
    sample!(1:prob.numdata,method.ind,replace=false);
    method.S[:] = x-method.prevx;# stores the difference
    d[:] =-prob.g_eval(x,method.ind)+prob.g_eval(method.prevx,method.ind)+prob.Hess_D(method.prevx,method.ind).*method.S  -
    method.grad -method.H.*method.S;
  end
  if(options.precondition) #println("qN")
    d[:] =(method.H).^(-1).*d;
  end

end
