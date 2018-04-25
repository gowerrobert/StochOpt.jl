function descent_SAGA(x::Array{Float64},prob::Prob,options::MyOptions,method::SAGAmethod, N::Int64,d::Array{Float64})

  s = sample(1:prob.numdata,options.batchsize,replace=false);
  if(method.minibatch_type == "rade") #Take the averaged of the gradients then broadcast to the columns of Jac
    method.aux[:]  = -sum(method.Jac[:,s],2);
    method.gi[:] = prob.g_eval(x,s);
    method.aux[:]  +=  options.batchsize*method.gi; # calculating the update vector   (DF^k-J^k) Proj 1
    method.Jac[:,s] .=  method.gi;
  else  #Assign each gradient to a different column of Jac
    method.aux[:]  =  -sum(method.Jac[:,s],2); # calculating the update vector   (DF^k-J^k) Proj 1
    prob.Jac_eval!(x,s,method.Jac);
    method.aux[:]  += sum(method.Jac[:,s],2);
  end

  #update SAG estimate:   1/n J^{k+1}1 = 1/n J^k 1 + 1/n (DF^k-J^k) Proj 1
  method.SAGgrad[:] = method.SAGgrad + (1/prob.numdata)*method.aux;
  # Gradient estimate
  if(method.unbiased)
    d[:] = -method.SAGgrad- (1/options.batchsize)*method.aux;
  else
    d[:] = -method.SAGgrad; #- (1/options.batchsize)*method.aux;
  end
end
