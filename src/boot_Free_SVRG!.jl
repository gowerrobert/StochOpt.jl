"""
    initiate_Free_SVRG(prob, options ; numinneriters=0, averaged_reference_point=false)

Initiate the Free-SVRG method for b-nice sampling.
It uniformly picks b data points out of n at each iteration to build an estimate of the gradient.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
- **Sampling** sampling: sampling object (b-nice or independent sampling)
- **Int64** numinneriters: size of the inner loop (theoretical value m^* if set to -1, number of data samples n if set to 0)
- **Bool** averaged_reference_point: select if the reference point is an average of the iterates of the inner loop or the last one
# OUTPUTS
- **Free\\_SVRG\\_method** method: Free-SVRG method created by `initiate_Free_SVRG`

# REFERENCES
__Our Title__\\
Francis Bach, Othmane Sebbouh, Robert M. Gower and Nidham Gazagnadou\\
arXiv:??????, 2019
"""
function initiate_Free_SVRG(prob::Prob, options::MyOptions, sampling::Sampling ; numinneriters::Int64=0, averaged_reference_point::Bool=false)
    n = prob.numdata

    ## No deterministic number of computed gradients per iteration because of the inner and outer loop scheme
    b = sampling.batchsize # deterministic or average mini-batch size
    epocsperiter = 0
    gradsperiter = 0
    number_computed_gradients = Int64[0] # dynamic table of the number of computed gradients at each iteration

    name = string("Free-SVRG-", sampling.name)

    stepmethod = descent_Free_SVRG!
    bootmethod = boot_Free_SVRG!
    reset = reset_Free_SVRG!

    stepsize = 0.0

    L = prob.L
    Lmax = prob.Lmax
    mu = prob.mu

    expected_smoothness = ((n-b)/(b*(n-1)))*Lmax + ((n*(b-1))/(b*(n-1)))*L
    expected_residual = ((n-b)/(b*(n-1)))*Lmax

    # if numinneriters == -1
    #     if occursin("nice", sampling.name)
    #         numinneriters = floor(Int, (expected_smoothness + 2*expected_residual) / mu) # theoretical optimal value for b-nice sampling
    #     else
    #         error("No theoretical inner loop size available for Free-SVRG with this sampling")
    #     end
    # elseif numinneriters < -1 || numinneriters == 0
    #     error("Invalid inner loop size")
    # end

    if numinneriters < 1
        error("Invalid inner loop size")
    end

    reference_point = zeros(prob.numfeatures)
    new_reference_point = zeros(prob.numfeatures)
    reference_grad = zeros(prob.numfeatures)
    if averaged_reference_point
        averaging_weights = zeros(numinneriters)
    else
        averaging_weights = Float64[]
    end

    method = Free_SVRG_method(epocsperiter, gradsperiter, number_computed_gradients, name, stepmethod, bootmethod, b, stepsize, L, Lmax, mu, expected_smoothness, expected_residual, numinneriters, reference_point, new_reference_point, reference_grad, averaging_weights, reset, sampling)

    return method
end


"""
    boot_Free_SVRG!(prob, method, options)

Modify the method to set the stepsize based on the smoothness constants of the problem stored in **Free\\_SVRG\\_method** and possibly sets the number of skipped error calculation if not specfied such that 30 points are to be plotted.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **Free\\_SVRG\\_method** method: Free-SVRG method created by `initiate_Free_SVRG`
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
# OUTPUTS
- **NONE**
"""
function boot_Free_SVRG!(prob::Prob, method::Free_SVRG_method, options::MyOptions)
    if options.stepsize_multiplier > 0.0
        println("Manually set step size")
        method.stepsize = options.stepsize_multiplier
    elseif options.stepsize_multiplier == -1.0
        if occursin("nice", method.sampling.name)
            method.stepsize = 1/(2*(method.expected_smoothness + 2*method.expected_residual)) # theoretical optimal value for b-nice sampling
            println("Automatically set Free-SVRG step size: ", method.stepsize)
        else
            error("No theoretical step size available for Free-SVRG with this sampling")
        end
    else
        error("Invalid options.stepsize_multiplier")
    end

    if !isempty(method.averaging_weights)
        averaging_weights = [(1-method.stepsize*method.mu)^(method.numinneriters-1-t) for t in 0:(method.numinneriters-1)]
        method.averaging_weights = averaging_weights ./ sum(averaging_weights)

        println("Sanity check: weights sum to 1 ? ---> ", sum(method.averaging_weights),"\n\n")
    end
    # println("Averaging weights")
    # println(method.averaging_weights)

    # WARNING: The following if statement does not seem to modify the method that is returned afterwards...
    if options.skip_error_calculation == 0.0
        options.skip_error_calculation = ceil(options.max_epocs*prob.numdata/(options.batchsize*30)) # show 30 points between 0 and the max number of epochs
        # 20 points over options.max_epocs when there are options.max_epocs *prob.numdata/(options.batchsize)) iterates in total
    end
    println("Skipping ", options.skip_error_calculation, " iterations per epoch\n")
end


"""
    reset_Free_SVRG(prob, method, options)

Reset the Free-SVRG method with b-nice sampling, especially the step size, the point and gradient reference.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **Free\\_SVRG\\_method** method: Free-SVRG method created by `initiate_Free_SVRG`
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
# OUTPUTS
- **NONE**
"""
function reset_Free_SVRG!(prob::Prob, method::Free_SVRG_method, options::MyOptions)
    println("\n---- RESET FREE-SVRG ----\n")

    method.number_computed_gradients = Int64[0]
    method.stepsize = 0.0 # Will be set during boot

    method.reference_point = zeros(prob.numfeatures)
    method.reference_grad = zeros(prob.numfeatures)
    if !isempty(method.averaging_weights)
        averaging_weights = zeros(method.numinneriters)
    else
        method.averaging_weights = Float64[]
    end
end