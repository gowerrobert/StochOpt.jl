function descent_SAGApartition(x::Array{Float64},prob::Prob,options::MyOptions,sg::SAGAmethod, N::Int64,d::Array{Float64})

  Smini = size(sg.minibatches);
  if(sg.probability_type == "")
    i = sample(1:Smini[1],1,replace=false);
  else
    i =  wsample(1:Smini[1], sg.probs);
  end
  i = i[1];
  s = vec(sg.minibatches[i,:]);

  ## Code for full Jacobian, i.e, will work even when it's not a linear model
  # sg.aux[:]  = -sg.Jac[:,i];
  # sg.gi[:] = prob.g_eval(x,s);
  # sg.aux[:]  +=  options.batchsize*sg.gi; # calculating the update vector   (DF^k-J^k) Proj 1
  # sg.Jac[:,i] .=  options.batchsize*sg.gi;

  # no method matching
  # logistic_scalar_grad(::SparseVector{Float64,Int64}, ::Float64, ::Array{Float64,1})
  # Closest candidates are:
  # logistic_scalar_grad(::Any, !Matched::Array{Float64,N}, ::Array{Float64,N}) at /home/rgower/Dropbox/Software/StochOpt/test/../src/logistic_grad.jl:11

  # t = logistic_phi(prob.y[s].*(prob.X[:,s])'*x);
  # scalargrad =prob.y[s].*(t.- 1);
  scalargrad = prob.scalar_grad_eval(x,s);
  sg.gi[:] = (prob.X[:,s])*scalargrad;
  sg.aux[:]  = -(prob.X[:,s])*sg.Jac[s];
  sg.aux[:]  +=  sg.gi; # calculating the update vector   (DF^k-J^k) Proj 1
  sg.Jac[s]  =  scalargrad;

  #update SAG estimate:   1/n J^{k+1}1 = 1/n J^k 1 + 1/n (DF^k-J^k) Proj 1
  sg.SAGgrad[:] = sg.SAGgrad + (1.0/prob.numdata)*sg.aux;
  #unbiased gradient estimate
  if(sg.unbiased)
    d[:] = -sg.SAGgrad- (1.0/(prob.numdata*(sg.probs[i])))*sg.aux;
  else
    d[:] = -sg.SAGgrad; #- (1/options.batchsize)*sg.aux;
  end
  d[:] =  d - prob.lambda.*x; # Add on the the regularization gradient
end

function descent_SAGA_adapt(x::Array{Float64},prob::Prob,options::MyOptions,sg::SAGAmethod, N::Int64,d::Array{Float64})
   if(N%(5*prob.numdata) == 0) # Every xx passes over the data, reset probabilities
      println("N: $(N), 5*prob.numdata:  $(5*prob.numdata),  update probs!")
      local_pi_estimate!(x, prob,options,  sg);
      ## Update stepsize
   end
   descent_SAGApartition(x,prob,options,sg, N,d);
end

function descent_SAGA_adapt2(x::Array{Float64},prob::Prob,options::MyOptions,sg::SAGAmethod, N::Int64,d::Array{Float64})

    i =  wsample(1:prob.numdata , sg.probs);
    # t = logistic_phi(prob.y[i].*(prob.X[:,i])'*x);
    # t = t[1];
    # scalargrad =prob.y[i].*(t.- 1);
    scalargrad, scalarhess = prob.scalar_grad_hess_eval(x,[i]);
    # sg.gi[:] = (prob.X[:,i])*scalargrad.+(prob.lambda).*x;
    scalargrad = scalargrad[1];
    # display("scalargrad :")
    # display(scalargrad)
    sg.gi[:] = (prob.X[:,i])*scalargrad;
    # sg.aux[:]  = -sg.Jac[:,i];  # When the function is not a liner model Jac is the full n X d Jacobian
    sg.aux[:]  = -(prob.X[:,i])*sg.Jac[i];
    sg.aux[:]  +=  sg.gi; # calculating the update vector   (DF^k-J^k) Proj 1
    # sg.Jac[:,i] .=  sg.gi;  # When the function is not a liner model Jac is the full n X d Jacobian
    sg.Jac[i]  =  scalargrad;

    #update SAG estimate:   1/n J^{k+1}1 = 1/n J^k 1 + 1/n (DF^k-J^k) Proj 1
    sg.SAGgrad[:] = sg.SAGgrad + (1/prob.numdata)*sg.aux;
    #unbiased gradient estimate
    if(sg.unbiased)
      d[:] = -sg.SAGgrad- (1/(prob.numdata*sg.probs[i]))*sg.aux;
    else
      d[:] = -sg.SAGgrad; #- (1/options.batchsize)*sg.aux;
    end
    d[:] =  d - prob.lambda.*x; # Add on the the regularization gradient

    candidateprob = 4*(scalarhess[1]*norm(prob.X[:,i])^2+prob.lambda) + prob.numdata*sg.mu;
    sg.probs[:] = sg.Z*sg.probs;
    # ## removing probs[i] previous contribution
    sg.Z = sg.Z -  sg.probs[i];
    sg.probs[i] = max(sg.probs[i], candidateprob);
    sg.Z = sg.Z +  sg.probs[i];
    sg.probs[:] = sg.probs/sg.Z;
    sg.stepsize = options.stepsize_multiplier*(prob.numdata/sg.Z);
end

function local_pi_estimate!(w::Array{Float64}, prob::Prob,options::MyOptions,  sg::SAGAmethod)
  Xx  = prob.X'*w;
  yXx = prob.y.*Xx;
  sg.probs[:] =logistic_phi(yXx) ;
  sg.probs[:] = sg.probs.*(1-sg.probs);
  # recording the max phi
  sg.phis[:] = max.(sg.probs, sg.phis);
  println("mean phi val: ", mean(sg.phis) )
  # sg.probs[:] = sqrt(sg.probs);
  sg.probs[:] = sg.phis.*vec(sum(prob.X.^2,1))+prob.lambda; # local Li constants
  # munew = mean(sg.probs)+prob.lambda;
  # println("with phi part: ", mean(sg.probs) )
  # eigmax(Symmetric(full(prob.X[:,C]'*prob.X[:,C])))/length(C) +prob.lambda;
  sg.probs[:] = sg.probs*4.+prob.numdata*sg.mu;
  stepsizeinverse = mean(sg.probs);#*prob.numfeatures;
  sg.probs[:] = sg.probs./sum(sg.probs);
  # update stepsize
  # println("previous stepsize: ", sg.stepsize )
  sg.stepsize = options.stepsize_multiplier/stepsizeinverse;
  # println("new stepsize: ", options.stepsize_multiplier/stepsizeinverse )
end


    ## Junk
    # sg.probs[:] = sg.probs.*(1-sg.probs);
    # updating  Phi_i

        # println("Z sum: ", sg.Z)
    # sg.phis[i] = max.(t*(1-t), sg.phis[i]);
    # sg.probs[:] = sg.probs*sg.phis[end];
    # ## removing probs[i] previous contribution
    # sg.phis[end] = sg.phis[end] - 4*sg.probs[i] ;
    # stepsizeinverse = (options.stepsize_multiplier*prob.numdata)/sg.stepsize;
    # stepsizeinverse = stepsizeinverse -sg.probs[i];
    # ## update probs[i]
    # sg.probs[i] = t*(1-t)*norm(prob.X[:,i])^2+prob.lambda + prob.numdata*sg.mu;
    # ## add back the contribution of probs[i]
    # sg.phis[end] = sg.phis[end] + 4*sg.probs[i] ;
    # sg.probs[:] = sg.probs/sg.phis[end];
    # stepsizeinverse = stepsizeinverse+sg.probs[i];
    # stepsizeinverse = stepsizeinverse/sg.phis[end];

    #  sg.phis.*vec(sum(prob.X.^2,1))+prob.lambda; # local Li constants
    # sg.probs[:] = sg.probs*4.+prob.numdata*sg.mu;



    # stepsizeinverse = mean(sg.probs);#*prob.numfeatures;
    # sg.probs[:] = sg.probs./sum(sg.probs);
    # update stepsize
    # println("previous stepsize: ", sg.stepsize )

   # if(N%(5*prob.numdata) == 0) # Every xx passes over the data, reset probabilities
   #    println("N: $(N), 5*prob.numdata:  $(5*prob.numdata),  update probs!")
   #    local_pi_estimate!(x, prob,options,  sg);
   #    ## Update stepsize
   # end
