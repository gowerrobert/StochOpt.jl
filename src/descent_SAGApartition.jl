function descent_SAGApartition(x::Array{Float64},prob::Prob,options::MyOptions,method::SAGAmethod, N::Int64,d::Array{Float64})

  Smini = size(method.minibatches);
  if(method.probability_type == "")
    i = sample(1:Smini[1],1,replace=false);
  else
    i =  wsample(1:Smini[1] , method.probs);
  end
  s = vec(method.minibatches[i,:])


  method.aux[:]  = -method.Jac[:,i];
  method.gi[:] = prob.g_eval(x,s);
  method.aux[:]  +=  options.batchsize*method.gi; # calculating the update vector   (DF^k-J^k) Proj 1
  method.Jac[:,i] .=  options.batchsize*method.gi;


  #update SAG estimate:   1/n J^{k+1}1 = 1/n J^k 1 + 1/n (DF^k-J^k) Proj 1
  method.SAGgrad[:] = method.SAGgrad + (1/prob.numdata)*method.aux;
  #unbiased gradient estimate
  if(method.unbiased)
    d[:] = -method.SAGgrad- (1/(prob.numdata*method.probs[i]))*method.aux;
  else
    d[:] = -method.SAGgrad; #- (1/options.batchsize)*method.aux;
  end
end
