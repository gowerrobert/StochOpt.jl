"""
    initiate_SAGA_nice(prob::Prob, options::MyOptions; unbiased=true)

Initiate the SAGA method for ``τ``--nice sampling.
It uniformly picks ``τ`` data points out of ``n`` at each iteration to build an estimate of the gradient.

#INPUTS:\\
    - prob: considered problem (i.e. logistic regression, ridge ression...) of the type **Prob** (see src/StochOpt.jl)\\
    - options: different options such as the mini-batch size, the stepsize_multiplier etc of the type MyOptions (see src/StochOpt.jl)\\
    - unbiased: select the desired estimate of the gradient. If `true`, SAGA is implemented, else if `false` SAG is implemented\\
#OUTPUTS:\\
    - SAGA_nice_method: SAGA mini-batch method for ``τ``--nice sampling of type SAGA_nice_method (see src/StochOpt.jl)
"""
function initiate_SAGA_nice(prob::Prob, options::MyOptions; unbiased=true)
    # options.stepsize_multiplier = 1; # WHY ?!
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
    name = string(name, "-nice");

    descent_method = descent_SAGA_nice;

    minibatches = [];
    Jac = zeros(prob.numfeatures, prob.numdata); # Jacobian of size d x n
    Jacsp = spzeros(1); #spzeros(prob.numfeatures, prob.numdata);
    SAGgrad = zeros(prob.numfeatures);
    gi = zeros(prob.numfeatures);
    aux = zeros(prob.numfeatures);
    stepsize = 0.0;
    probs = [];
    Z = 0.0;

    return SAGA_nice_method(epocsperiter, gradsperiter, name, descent_method, boot_SAGA_nice, minibatches, unbiased,
                            Jac, Jacsp, SAGgrad, gi, aux, stepsize, probs, Z, reset_SAGA_nice);
end


"""
    boot_SAGA_nice(prob::Prob, method, options::MyOptions)

Set the stepsize based on the smoothness constants of the problem stored in **SAGA_nice_method**.

#INPUTS:\\
    - prob: considered problem (i.e. logistic regression, ridge regression...) of the type **Prob** (see src/StochOpt.jl)\\
    - method: **SAGA_nice_method** created by `initiate_SAGA_nice` \\
    - options: different options such as the mini-batch size, the stepsize_multiplier etc of the type MyOptions (see src/StochOpt.jl)\\
#OUTPUTS:\\
    - SAGA_nice_method: SAGA mini-batch method for ``τ``--nice sampling of type SAGA_nice_method (see src/StochOpt.jl)
"""
function boot_SAGA_nice(prob::Prob, method, options::MyOptions)
    # /!\ WARNING: this function modifies its own arguments (`method` and `options`) and returns method! Shouldn't we name it "boot_SAGA_nice!(...)" with an "!" ?
    tau = options.batchsize;
    n = prob.numdata;
    L = prob.L;
    Lmax = prob.Lmax;
    Lbar = prob.Lbar;

    if(occursin("lgstc", prob.name)) # julia 0.7
        ## Correcting for logistic since phi'' <= 1/4
        # Lexpected = Lexpected/4;
        L /= 4;
        Lmax /= 4;
        Lbar /= 4;
    end
    leftcoeff = (n*(tau-1))/(tau*(n-1));
    rightcoeff = (n-tau)/(tau*(n-1));
    Lheuristic = leftcoeff*L + rightcoeff*Lmax;
    Lsimple = leftcoeff*Lbar + rightcoeff*Lmax;
    Lbernstein = 2*leftcoeff*L + (1/tau)*((n-tau)/(n-1) + (4/3)*log(prob.numfeatures))*Lmax;
    rightterm = ((n-tau)/(tau*(n-1)))*Lmax + (prob.mu*n)/(4*tau); # Right-hand side term in the max in the denominator

    if options.stepsize_multiplier == -1.0
        ## Heuristic
        println("Heuristic step size");
        sleep(2);
        method.stepsize = 1.0/(4.0*max(Lheuristic, rightterm));
    elseif options.stepsize_multiplier == -2.0
        ## Simple bound
        println("Simple step size");
        sleep(2);
        method.stepsize = 1.0/(4.0*max(Lsimple, rightterm));
    elseif options.stepsize_multiplier == -3.0
        ## Bernstein bound
        println("Bernstein step size");
        sleep(2);
        method.stepsize =  1.0/(4.0*max(Lbernstein, rightterm));
    elseif options.stepsize_multiplier > 0.0 # Put it first and do computations only if necessary
        println("Manually set step size");
        method.stepsize = options.stepsize_multiplier;
    else
        error("Invalid options.stepsize_multiplier");
    end

    # WARNING: The following if statement does not seem to modify the method that is returned afterwards...
    if(options.skip_error_calculation == 0.0)
        options.skip_error_calculation = ceil(options.max_epocs*prob.numdata/(options.batchsize*30)); # show 30 points between 0 and the max number of epochs
        # 20 points over options.max_epocs when there are options.max_epocs *prob.numdata/(options.batchsize)) iterates in total
    end
    println("Skipping ", options.skip_error_calculation, " iterations per epoch")
    return method;
end






"""
    reset_SAGA_nice(prob::Prob, method, options::MyOptions)

Reset the SAGA method with  ``τ``--nice sampling, especially the step size, the gradient and the Jacobian estimates.

#INPUTS:\\
    - prob: considered problem (i.e. logistic regression, ridge regression...) of the type **Prob** (see src/StochOpt.jl)\\
    - method: **SAGA_nice_method** created by `initiate_SAGA_nice` \\
    - options: different options such as the mini-batch size, the stepsize_multiplier etc of the type MyOptions (see src/StochOpt.jl)\\
#OUTPUTS:\\
    - SAGA_nice_method: SAGA mini-batch method for ``τ``--nice sampling of type SAGA_nice_method (see src/StochOpt.jl)
"""
function reset_SAGA_nice(prob::Prob, method, options::MyOptions; unbiased=true)
    println("\n---- RESET SAGA NICE ----\n");

    # method.epocsperiter = options.batchsize/prob.numdata;
    # method.gradsperiter = options.batchsize;
    # method.unbiased = unbiased;
    # if unbiased
    #     name = "SAGA"
    # else
    #     name = "SAG"
    # end
    # if(options.batchsize > 1)
    #     name = string(name, "-", options.batchsize);
    # end
    # method.name = string(name, "-nice");

    method.minibatches = [];
    method.Jac = zeros(prob.numfeatures, prob.numdata); # Jacobian of size d x n
    method.Jacsp = spzeros(1);
    method.SAGgrad = zeros(prob.numfeatures);
    method.gi = zeros(prob.numfeatures);
    method.aux = zeros(prob.numfeatures);
    method.stepsize = 0.0;
    method.probs = [];
    method.Z = 0.0;

    return method
end