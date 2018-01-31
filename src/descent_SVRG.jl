function descent_SVRG(x::Array{Float64},prob::Prob,options::MyOptions,method::Method, N::Int64,d::Array{Float64})
  # SVRG outerloop
  if(N%method.numinneriters ==1 || method.numinneriters ==1)# Reset reference point, grad estimate and Hessian estimate
  #  println("SVRG outer loop at iteration: ",N)
    method.prevx[:]  = x;
    method.grad[:] = prob.g_eval(x,1:prob.numdata);
    d[:]= -method.grad;
  else
  sample!(1:prob.numdata,method.ind,replace=false);
  #  s = sample!(1:prob.numdata,options.batchsize,replace=false);
    # SVRG inner step
    d[:] = -prob.g_eval(x,method.ind)+prob.g_eval(method.prevx,method.ind)  -method.grad
  end
#    println("|d| ",norm(d))
end
