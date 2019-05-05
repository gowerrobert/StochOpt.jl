"""
    initiate_SVRG_nice(prob::Prob, options::MyOptions; unbiased=true)

Initiate the SVRG method for b-nice sampling.
It uniformly picks b data points out of n at each iteration to build an estimate of the gradient.

# INPUTS:
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...\\
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...\\
# OUTPUTS:
- **SVRG\\_nice\\_method** method: SVRG mini-batch method for b-nice sampling
"""
function initiate_SVRG_nice(prob::Prob, options::MyOptions)
    epocsperiter = options.batchsize/prob.numdata;
    gradsperiter = options.batchsize;
    name = "SVRG"
    if options.batchsize > 1
        name = string(name, "-", options.batchsize);
    end
    name = string(name, "-nice");

    stepmethod = descent_SVRG_nice!;
    bootmethod = boot_SVRG_nice!;

    minibatch_size = options.batchsize;
    stepsize = 0.0;
    probs = [];
    Z = 0.0;

    L = prob.L;
    Lmax = prob.Lmax;
    mu = prob.mu;

    n = prob.numdata;
    b = minibatch_size;
    expected_smoothness = ((n-b)/(b*(n-1)))*Lmax + ((n*(b-1))/(b*(n-1)))*L;
    expected_residual = ((n-b)/(b*(n-1)))*Lmax;

    numinneriters = 10; #prob.numdata;
    reference_point = zeros(prob.numfeatures);
    reference_grad = zeros(prob.numfeatures);
    averaging_weights = [];

    method = SVRG_nice_method(epocsperiter, gradsperiter, name, stepmethod, bootmethod, minibatch_size, stepsize, probs, Z, L, Lmax, mu, expected_smoothness, expected_residual, numinneriters, reference_point, reference_grad, averaging_weights, reset_SVRG_nice!);

    return method
end


"""
    boot_SVRG_nice!(prob::Prob, method, options::MyOptions)

Modify the method to set the stepsize based on the smoothness constants of the problem stored in **SVRG\\_nice\\_method** and possibly sets the number of skipped error calculation if not specfied such that 30 points are to be plotted.

# INPUTS:
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...\\
- **SVRG\\_nice\\_method** method: SVRG nice method created by `initiate_SVRG_nice`\\
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...\\
"""
function boot_SVRG_nice!(prob::Prob, method, options::MyOptions)
    if options.stepsize_multiplier > 0.0
        println("Manually set step size");
        method.stepsize = options.stepsize_multiplier;
    elseif options.stepsize_multiplier == -1.0
        println("Automatically set Free-SVRG step size");
        method.stepsize = 1/(2*(method.expected_smoothness + 2*method.expected_residual));
        println("Theoretical step size: ", SVRG_nice.stepsize);
    else
        error("Invalid options.stepsize_multiplier");
    end

    averaging_weights = [(1-method.stepsize*method.mu)^(method.numinneriters-1-t) for t in 0:(method.numinneriters-1)];
    method.averaging_weights = averaging_weights ./ sum(averaging_weights);
    println("Theoretical averaging weights");
    println(method.averaging_weights);

    # WARNING: The following if statement does not seem to modify the method that is returned afterwards...
    if options.skip_error_calculation == 0.0
        options.skip_error_calculation = ceil(options.max_epocs*prob.numdata/(options.batchsize*30)); # show 30 points between 0 and the max number of epochs
        # 20 points over options.max_epocs when there are options.max_epocs *prob.numdata/(options.batchsize)) iterates in total
    end
    println("Skipping ", options.skip_error_calculation, " iterations per epoch\n");
end


"""
    reset_SVRG_nice(prob::Prob, method, options::MyOptions)

Reset the SVRG method with b-nice sampling, especially the step size, the point and gradient reference.

# INPUTS:
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...\\
- **SVRG\\_nice\\_method**: SVRG mini-batch method for b-nice sampling\\
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...\\
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
end