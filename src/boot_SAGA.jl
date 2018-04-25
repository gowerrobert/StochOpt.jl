function intiate_SAGA( prob::Prob, options::MyOptions ; minibatch_type="nice", probability_type = "", unbiased = true)
# function for setting the parameters and initiating the SAGA method (and all it's variants)
  options.stepsize_multiplier =1;
  epocsperiter = options.batchsize/prob.numdata;
  gradsperiter = options.batchsize;
  if(unbiased)
    name = "SAGA"
  else
    name = "SAG"
  end
  if(options.batchsize>1)
    name = string(name,"-",options.batchsize);
  end
  minibatches =[];
  Jacsp = spzeros(1);#spzeros(prob.numfeatures,prob.numdata);
  SAGgrad = zeros(prob.numfeatures);
  gi = zeros(prob.numfeatures);
  aux = zeros(prob.numfeatures);
  descent_method = descent_SAGA;
  probs = [];
  mu = get_mu_str_conv(prob);

  if(minibatch_type =="partition")
    # L = mean(sum(prob.X.^2,1));
    Jac, minibatches, probs, name, L, Lmax = boot_SAGA_partition(prob,options,probability_type,name,mu);
    descent_method = descent_SAGApartition;
  else
    name = string(name,"-",minibatch_type);
    L = eigmax(full(prob.X*prob.X'))/prob.numdata+prob.lambda;
    Lmax = maximum(sum(prob.X.^2,1))+prob.lambda;
    Jac= zeros(prob.numfeatures, prob.numdata);
  end
  return SAGAmethod(epocsperiter, gradsperiter,name, descent_method, boot_SAGA, minibatches, minibatch_type, unbiased,
  Jac, Jacsp, SAGgrad, gi, aux,0.0, probs, probability_type, L, Lmax, mu);
end
#
function boot_SAGA(prob::Prob,method,options::MyOptions)

  tau = options.batchsize;
  n = prob.numdata;
  if(method.minibatch_type =="partition")  # 1/(4 L + n/tau mu)
    if(method.probability_type == "uni")
     Lexpected = method.Lmax;
    else
     Lexpected = method.L;
   end
  else # nice sampling     # interpolate Lmax and L
     Lexpected =   exp((1-tau)/( (n+0.00000001) -tau))*method.Lmax + ((tau-1)/(n-1))*method.L;
  end
  if (contains(prob.name,"lgstc"))
     Lexpected = Lexpected/4;    # THIS IS OFF by a factor of 2/3.
  end
  method.stepsize = options.stepsize_multiplier/(4*Lexpected+ (n/tau)*method.mu);
   if(options.skip_error_calculation ==0.0)
  options.skip_error_calculation =ceil(options.max_epocs*prob.numdata/(options.batchsize*20) ); # show 5 times per pass over the data
  # 20 points over options.max_epocs when there are options.max_epocs *prob.numdata/(options.batchsize)) iterates in total
   end
  println("Skipping ", options.skip_error_calculation, " iterations per epoch")
  return method;
end


function boot_SAGA_partition(prob::Prob,options::MyOptions, probability_type::AbstractString, name::AbstractString, mu::Float64)
  ## Setting up a partition mini-batch
  numpartitions = convert(Int64, ceil(prob.numdata/options.batchsize)) ;
  Jac = zeros(prob.numfeatures, numpartitions);
  #shuffle the data and then cute in contigious minibatches
  datashuff = shuffle(1:1:prob.numdata);
  # Pad wtih randomly selected indices
  resize!(datashuff, numpartitions*options.batchsize);
  samplesize = numpartitions*options.batchsize - prob.numdata;
  s = sample(1:prob.numdata,samplesize,replace=false);
  datashuff[prob.numdata+1:end] = s;
  # look up repmat or reshape matrix
  minibatches = reshape(datashuff, numpartitions, options.batchsize);
  ## Setting the probabilities
  probs = zeros(1,numpartitions);
  name = string(name,"-",probability_type);
  for i =1:numpartitions      # Calculating the Li's assumming it's a phi'' < 1/4 like a logistic fuction
    probs[i] = get_LC(prob, minibatches[i,:]);
    # probs[i] = eigmax(Symmetric(prob.X[:,minibatches[i,:]]'*prob.X[:,minibatches[i,:]]))/options.batchsize +prob.lambda;
  end
    Lmax= maximum(probs);
    L = mean(probs);
  if(probability_type == "Li")
    probs[:] = probs./sum(probs);
  elseif(probability_type == "uni")
    probs[:] .= 1/numpartitions;
  else
    probs[:] = probs.*4.+numpartitions*mu;
    probs[:] = probs./sum(probs);
  end
  # probs[ = vec(probs);
  return Jac, minibatches, vec(probs), name, L, Lmax
end
