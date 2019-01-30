function initiate_SAGA_partition(prob::Prob, options::MyOptions; minibatch_type="partition", probability_type="uni", unbiased=true)
    # function for setting the parameters and initiating the SAGA method (and all it's variants)
    options.stepsize_multiplier = 1;
    epocsperiter = options.batchsize/prob.numdata;
    gradsperiter = options.batchsize;
    if(unbiased)
        name = "SAGA"
    else
        name = "SAG"
    end
    if(options.batchsize > 1)
        name = string(name, "-", options.batchsize);
    end
    minibatches = [];
    Jacsp = spzeros(1); #spzeros(prob.numfeatures, prob.numdata);
    SAGgrad = zeros(prob.numfeatures);
    gi = zeros(prob.numfeatures);
    aux = zeros(prob.numfeatures);
    descent_method = descent_SAGA_partition;
    probs = []; Z = 0.0; stepsize = 0.0;
    mu = prob.mu;

    if(probability_type == "ada")
        if(options.initial_point == "randn")
            x = randn(prob.numfeatures);
        elseif(options.initial_point == "zeros")
            println("adaSAGA doesn't work well with zeros initial point! Existing")
            return [];
        else
            println("FAIL: unknown initial point. Random initial point instead.")
            x = rand(prob.numfeatures);
        end
        Jac = zeros(prob.numdata);
        # Jac= zeros(prob.numfeatures, prob.numdata);
        # name = string(name, "-", probability_type);
        name = string(name, "-partition-", probability_type);
        yXx = prob.y.*(prob.X'*x); # initial random guess
        probs = logistic_phi(yXx) ;
        probs[:] = probs .*(1 .- probs);
        probs[:] = max.(probs, 0.25/8.0); ## IMPORTANT: this establishes a minimum value for phi''
        probs[:] = probs .* vec(sum(prob.X.^ 2, dims = 1)) .+ prob.lambda;
        Lmax = maximum(probs);
        L = mean(probs);
        probs[:] = probs.*4 .+ prob.numdata*mu;
        Z = sum(probs);
        probs[:] = probs ./ Z;
        stepsize = prob.numdata/Z;
        descent_method = descent_SAGA_adapt2;#descent_SAGA_adapt;
    elseif(probability_type == "uni"|| probability_type == "opt" || probability_type == "Li") # adding "Li" and "opt" cases
        Jac, minibatches, probs, name, L, Lmax, stepsize = boot_SAGA_partition(prob, options, probability_type, name, mu); # Does "ada" exist for "partition"?
        descent_method = descent_SAGA_partition;
    else
        error("unknown probability_type name (", probability_type, ").");
        # println("\nFAIL: unknown probability_type name (", probability_type, ")\n");
        # exit(1);
    end
    ### End partition
    return SAGAmethod(epocsperiter, gradsperiter, name, descent_method, boot_SAGA, minibatches, minibatch_type, unbiased,
    Jac, Jacsp, SAGgrad, gi, aux, stepsize, probs, probability_type, Z, L, Lmax, mu);
end

function boot_SAGA(prob::Prob, method, options::MyOptions)
    tau = options.batchsize;
    n = prob.numdata;

    if(method.probability_type == "uni")
        Lexpected = method.Lmax;
    else
        Lexpected = method.L;
    end
    # nice sampling     # interpolate Lmax and L
    Li_s = get_Li(prob.X, prob.lambda);
    Lbar = mean(Li_s);
    leftcoeff = (n*(tau-1))/(tau*(n-1));
    rightcoeff = (n-tau)/(tau*(n-1));
    simplebound = leftcoeff*Lbar + rightcoeff*method.Lmax;
    # Lexpected = exp((1 - tau)/((n + 0.1) - tau))*method.Lmax + ((tau - 1)/(n - 1))*method.L;
    if(occursin(prob.name, "lgstc"))
        Lexpected = Lexpected/4;    #  correcting for logistic since phi'' <= 1/4
    end
    rightterm = ((n-tau)/(tau*(n-1)))*method.Lmax + (method.mu*n)/(4*tau); # Right-hand side term in the max in the denominator
    ### Broken code: how is this possible? simplebound not defined depending on execution
    method.stepsize = 1.0/(4*max(simplebound, rightterm));
    # method.stepsize = options.stepsize_multiplier/(4*Lexpected + (n/tau)*method.mu);

    if(options.skip_error_calculation == 0.0)
        options.skip_error_calculation = ceil(options.max_epocs*prob.numdata/(options.batchsize*30)); # show 5 times per pass over the data
        # 20 points over options.max_epocs when there are options.max_epocs *prob.numdata/(options.batchsize)) iterates in total
    end
    println("Skipping ", options.skip_error_calculation, " iterations per epoch")
    return method;
end

function boot_SAGA_partition(prob::Prob, options::MyOptions, probability_type::AbstractString, name::AbstractString, mu::Float64)
    ## Setting up a partition mini-batch
    numpartitions = convert(Int64, ceil(prob.numdata/options.batchsize)) ;
    # Jac = zeros(prob.numfeatures, numpartitions);
    Jac = zeros(prob.numdata);
    #shuffle the data and then cute in contigious minibatches
    datashuff = shuffle(1:1:prob.numdata);
    # Pad wtih randomly selected indices
    resize!(datashuff, numpartitions*options.batchsize);
    samplesize = numpartitions*options.batchsize - prob.numdata;
    s = sample(1:prob.numdata, samplesize, replace=false);
    datashuff[prob.numdata+1:end] = s;
    # look up repmat or reshape matrix
    minibatches = reshape(datashuff, numpartitions, options.batchsize);
    ## Setting the probabilities
    probs = zeros(1, numpartitions);
    # name = string(name, "-", probability_type);
    name = string(name, "-partition-", probability_type);
    for i = 1:numpartitions # Calculating the Li's assumming it's a phi'' < 1. Remember for logistic fuction phi'' <1/4
        probs[i] = get_LC(prob.X, prob.lambda, minibatches[i, :]);
        # probs[i] = eigmax(Symmetric(prob.X[:,minibatches[i,:]]'*prob.X[:,minibatches[i,:]]))/options.batchsize +prob.lambda;
    end
    Lmax = maximum(probs);
    L = mean(probs);
    if(probability_type == "Li")
        probs[:] = probs./sum(probs);
        stepsize = 1/(4*L+numpartitions*mu);
    elseif(probability_type == "uni")
        probs[:] .= 1/numpartitions;
        stepsize = 1/(4*Lmax+numpartitions*mu);
    elseif(probability_type == "opt") # What is else for? I think it is "opt"
        probs[:] = probs.*4 .+numpartitions*mu; # julia 0.7
        stepsize = 1/(mean(probs));
        probs[:] = probs./sum(probs);
    else
        error("unknown probability_type name (", probability_type, ").")
        # println("\nFAIL: unknown probability_type name (", probability_type, ")\n");
        # exit(1);
    end
    # probs[ = vec(probs);
    return Jac, minibatches, vec(probs), name, L, Lmax, stepsize
end
