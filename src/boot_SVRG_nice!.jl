"""
    initiate_SVRG_nice(prob::Prob, options::MyOptions; unbiased=true)

Initiate the SVRG method for b-nice sampling.
It uniformly picks b data points out of n at each iteration to build an estimate of the gradient.

#INPUTS:\\
    - prob: considered problem (i.e. logistic regression, ridge ression...) of the type **Prob** (see src/StochOpt.jl)\\
    - options: different options such as the mini-batch size, the stepsize_multiplier etc of the type MyOptions (see src/StochOpt.jl)\\
#OUTPUTS:\\
    - SVRG_nice_method: SVRG mini-batch method for b-nice sampling of type SVRG_nice_method (see src/StochOpt.jl)
"""
function initiate_SVRG_nice(prob::Prob, options::MyOptions)
    epocsperiter = options.batchsize/prob.numdata;
    gradsperiter = options.batchsize;
    name = "SVRG"
    if options.batchsize > 1
        name = string(name, "-", options.batchsize);
    end
    name = string(name, "-nice");

    descent_method = descent_SVRG_nice!;

    minibatch_size = 0;
    stepsize = 0.0;
    probs = [];
    Z = 0.0;

    L = 0.0;
    Lmax = 0.0;
    mu = 0.0;
    expected_smoothness = 0.0;
    expected_residual = 0.0;

    numinneriters = 10;
    reference_point = zeros(prob.numfeatures);
    reference_grad = zeros(prob.numfeatures);
    averaging_weights = [];

    return SVRG_nice_method(epocsperiter, gradsperiter, name, descent_method, boot_SVRG_nice!, minibatch_size, stepsize, probs, Z, L, Lmax, mu, expected_smoothness, expected_residual, numinneriters, reference_point, reference_grad, averaging_weights, reset_SVRG_nice!);
end


"""
    boot_SVRG_nice!(prob::Prob, method, options::MyOptions)

Set the stepsize based on the smoothness constants of the problem stored in **SVRG_nice_method**.

#INPUTS:\\
    - prob: considered problem (e.g., logistic regression, ridge regression...) of the type **Prob** (see src/StochOpt.jl)\\
    - method: **SVRG_nice_method** created by `initiate_SVRG_nice` \\
    - options: different options such as the mini-batch size, the stepsize_multiplier etc of the type MyOptions (see src/StochOpt.jl)\\
#OUTPUTS:\\
    - SVRG_nice_method: SVRG mini-batch method for b-nice sampling of type SVRG_nice_method (see src/StochOpt.jl)
"""
function boot_SVRG_nice!(prob::Prob, method, options::MyOptions)
    if options.stepsize_multiplier > 0.0 # Put it first and do computations only if necessary
        println("Manually set step size");
        method.stepsize = options.stepsize_multiplier;
    else
        error("Invalid options.stepsize_multiplier");
    end

    # WARNING: The following if statement does not seem to modify the method that is returned afterwards...
    if options.skip_error_calculation == 0.0
        options.skip_error_calculation = ceil(options.max_epocs*prob.numdata/(options.batchsize*30)); # show 30 points between 0 and the max number of epochs
        # 20 points over options.max_epocs when there are options.max_epocs *prob.numdata/(options.batchsize)) iterates in total
    end
    println("Skipping ", options.skip_error_calculation, " iterations per epoch")
    # return method
end


"""
    reset_SVRG_nice(prob::Prob, method, options::MyOptions)

Reset the SVRG method with  b-nice sampling, especially the step size, the gradient and the Jacobian estimates.

#INPUTS:\\
    - prob: considered problem (e.g., logistic regression, ridge regression...) of the type **Prob** (see src/StochOpt.jl)\\
    - method: **SVRG_nice_method** created by `initiate_SVRG_nice` \\
    - options: different options such as the mini-batch size, the stepsize_multiplier etc of the type MyOptions (see src/StochOpt.jl)\\
#OUTPUTS:\\
    - SVRG_nice_method: SVRG mini-batch method for b-nice sampling of type SVRG_nice_method (see src/StochOpt.jl)
"""
function reset_SVRG_nice!(prob::Prob, method, options::MyOptions)
    println("\n---- RESET SVRG NICE ----\n");

    method.minibatch_size = 0;
    method.stepsize = 0.0;
    method.probs = [];
    method.Z = 0.0;

    method.L = 0.0;
    method.Lmax = 0.0;
    method.mu = 0.0;
    method.expected_smoothness = 0.0;
    method.expected_residual = 0.0;

    method.numinneriters = 0;
    method.reference_point = zeros(prob.numfeatures);
    method.reference_grad = zeros(prob.numfeatures);
    method.averaging_weights = [];

    # return method # useless in a mutating function but SAGA_nice works like this
end