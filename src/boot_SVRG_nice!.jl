"""
    initiate_SVRG_nice(prob, options ; numinneriters=0)

Initiate the original SVRG method with option I for b-nice sampling.
It uniformly picks b data points out of n at each iteration to build an estimate of the gradient.

# INPUTS:
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
- **Int64** numinneriters: size of the inner loop (twice the number of data samples n if set to 0).
# OUTPUTS:
- **SVRG\\_nice\\_method** method: SVRG mini-batch method for b-nice sampling

# REFERENCES:
__Accelerating stochastic gradient descent using predictive variance reduction__\\
Rie Johnson and Tong Zhang\\
Advances in neural information processing systems. 2013.
"""
function initiate_SVRG_nice(prob::Prob, options::MyOptions ; numinneriters::Int64=0)
    epocsperiter = options.batchsize/prob.numdata
    gradsperiter = options.batchsize
    name = "SVRG"
    if options.batchsize > 1
        name = string(name, "-", options.batchsize)
    end
    name = string(name, "-nice")

    stepmethod = descent_SVRG_nice!
    bootmethod = boot_SVRG_nice!
    reset = reset_SVRG_nice!

    batchsize = options.batchsize
    stepsize = 0.0

    Lmax = prob.Lmax
    mu = prob.mu

    n = prob.numdata
    b = batchsize

    if numinneriters == 0
        numinneriters = 2*prob.numdata
    end
    reference_point = zeros(prob.numfeatures)
    reference_grad = zeros(prob.numfeatures)

    method = SVRG_nice_method(epocsperiter, gradsperiter, name, stepmethod, bootmethod, batchsize, stepsize, Lmax, mu, numinneriters, reference_point, reference_grad, reset)

    return method
end


"""
    boot_SVRG_nice!(prob, method, options)

Modify the method to set the stepsize based on the smoothness constants of the problem stored in **SVRG\\_nice\\_method** and possibly sets the number of skipped error calculation if not specfied such that 30 points are to be plotted.

# INPUTS:
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **SVRG\\_nice\\_method** method: SVRG nice method created by `initiate_SVRG_nice`
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
# OUTPUTS:
- **NONE**
"""
function boot_SVRG_nice!(prob::Prob, method::SVRG_nice_method, options::MyOptions)
    if options.stepsize_multiplier > 0.0
        println("Manually set step size")
        method.stepsize = options.stepsize_multiplier
    elseif options.stepsize_multiplier == -1.0
        println("Automatically set SVRG step size")
        method.stepsize = 1/(10*method.Lmax)
        options.stepsize_multiplier = method.stepsize # /!\ Modifies the options
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
    reset_SVRG_nice(prob, method, options)

Reset the SVRG method with b-nice sampling, especially the step size, the point and gradient reference.

# INPUTS:
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **SVRG\\_nice\\_method**: SVRG mini-batch method for b-nice sampling
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
# OUTPUTS:
- **NONE**
"""
function reset_SVRG_nice!(prob::Prob, method::SVRG_nice_method, options::MyOptions)
    println("\n---- RESET SVRG NICE ----\n")

    method.batchsize = options.batchsize
    method.stepsize = options.stepsize_multiplier

    method.reference_point = zeros(prob.numfeatures)
    method.reference_grad = zeros(prob.numfeatures)
end