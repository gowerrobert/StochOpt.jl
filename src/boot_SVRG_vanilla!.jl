"""
    initiate_SVRG_vanilla(prob, options ; numinneriters=0, probs=[])

Initiate the original SVRG method with option I, for b-nice or independent sampling.
It uniformly picks b data points out of n at each iteration to build an estimate of the gradient.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
- **Sampling** sampling: sampling object (b-nice or independent sampling)
- **Int64** numinneriters: size of the inner loop (twice the number of data samples n if set to 0)
# OUTPUTS
- **SVRG\\_vanilla\\_method** method: original SVRG method (option I) created by `initiate_SVRG_vanilla`

# REFERENCES
__Accelerating stochastic gradient descent using predictive variance reduction__\\
Rie Johnson and Tong Zhang\\
Advances in neural information processing systems. 2013.
"""
function initiate_SVRG_vanilla(prob::Prob, options::MyOptions, sampling::Sampling ; numinneriters::Int64=0)
    n = prob.numdata

    ## No deterministic number of computed gradients per iteration because of the inner and outer loop scheme
    b = sampling.batchsize # deterministic or average mini-batch size
    epocsperiter = 0
    gradsperiter = 0
    number_computed_gradients = 0 # dynamic counter of computed gradients

    name = string("SVRG-vanilla-", sampling.name)

    stepmethod = descent_SVRG_vanilla!
    bootmethod = boot_SVRG_vanilla!
    reset = reset_SVRG_vanilla!

    stepsize = 0.0

    Lmax = prob.Lmax
    mu = prob.mu

    if numinneriters == -1
        error("No theoretical inner loop size available for SVRG vanilla with this sampling") # no theoretical value given by Johnson & Zhang
        # Heuristic proposed is: m = 2*n
    elseif numinneriters < -1 || numinneriters == 0
        error("Invalid inner loop size")
    end

    reference_point = zeros(prob.numfeatures)
    reference_grad = zeros(prob.numfeatures)

    method = SVRG_vanilla_method(epocsperiter, gradsperiter, number_computed_gradients, name, stepmethod, bootmethod, b, stepsize, Lmax, mu, numinneriters, reference_point, reference_grad, reset, sampling)

    return method
end


"""
    boot_SVRG_vanilla!(prob, method, options)

Modify the method to set the stepsize based on the smoothness constants of the problem stored in **SVRG\\_vanilla\\_method** and possibly sets the number of skipped error calculation if not specfied such that 30 points are to be plotted.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **SVRG\\_vanilla\\_method** method: original SVRG method (option I) created by `initiate_SVRG_vanilla`
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
# OUTPUTS
- **NONE**
"""
function boot_SVRG_vanilla!(prob::Prob, method::SVRG_vanilla_method, options::MyOptions)
    if options.stepsize_multiplier > 0.0
        println("Manually set step size")
        method.stepsize = options.stepsize_multiplier
    elseif options.stepsize_multiplier == -1.0
        if method.sampling.name == "nice"
            method.stepsize = 1/(10*method.Lmax) # theoretical optimal value for 1-nice sampling by Johnson & Zhang
            println("Automatically set SVRG vanilla step size: ", method.stepsize)
        else
            error("No theoretical step size available for SVRG vanilla with this sampling")
        end

        println("Theoretical step size: ", method.stepsize)
    else
        error("Invalid options.stepsize_multiplier")
    end

    # WARNING: The following if statement does not seem to modify the method that is returned afterwards...
    if options.skip_error_calculation == 0.0
        options.skip_error_calculation = ceil(options.max_epocs*prob.numdata/(options.batchsize*30)) # show 30 points between 0 and the max number of epochs
        # 20 points over options.max_epocs when there are options.max_epocs *prob.numdata/(options.batchsize)) iterates in total
    end
    println("Skipping ", options.skip_error_calculation, " iterations per epoch\n")
end


"""
    reset_SVRG_vanilla(prob, method, options)

Reset the original SVRG method (option I), especially the step size, the point and gradient reference.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **SVRG\\_vanilla\\_method** method: original SVRG method (option I) created by `initiate_SVRG_vanilla`
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
# OUTPUTS
- **NONE**
"""
function reset_SVRG_vanilla!(prob::Prob, method::SVRG_vanilla_method, options::MyOptions)
    println("\n---- RESET SVRG VANILLA ----\n")

    method.number_computed_gradients = 0
    method.stepsize = 0.0 # Will be set during boot

    method.reference_point = zeros(prob.numfeatures)
    method.reference_grad = zeros(prob.numfeatures)
end