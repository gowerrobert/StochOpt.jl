"""
    initiate_Leap_SVRG(prob, options ; numinneriters=0, probs=[])

Initiate the Loopless-SVRG method.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
- **Sampling** sampling: sampling object (b-nice or independent sampling)
- **Float64** reference_update_proba: probability of updating the reference at each iteration of the (inner and only) loop (possible value: p = 1/n)
# OUTPUTS
- **Leap\\_SVRG\\_method** method: Loopless-SVRG method created by `initiate_Leap_SVRG`

# REFERENCES
__Our Title__\\
Othmane Sebbouh, Robert M. Gower and Nidham Gazagnadou\\
arXiv:??????, 2019
"""
function initiate_Leap_SVRG(prob::Prob, options::MyOptions, sampling::Sampling, reference_update_proba::Float64 ; stochastic_stepsize::Float64=0.0, gradient_stepsize::Float64=0.0)
    n = prob.numdata

    ## No deterministic number of computed gradients per iteration because of the inner and outer loop scheme
    b = sampling.batchsize # deterministic or average mini-batch size
    epocsperiter = 0
    gradsperiter = 0
    number_computed_gradients = Int64[0] # dynamic table of the number of computed gradients at each iteration

    name = string("Leap-SVRG-", sampling.name)

    stepmethod = descent_Leap_SVRG!
    bootmethod = boot_Leap_SVRG!
    reset = reset_Leap_SVRG!

    L = prob.L
    Lmax = prob.Lmax

    if occursin("nice", method.sampling.name)
        expected_smoothness = ((n-b)/(b*(n-1)))*Lmax + ((n*(b-1))/(b*(n-1)))*L
        expected_residual = ((n-b)/(b*(n-1)))*Lmax
    else
        error("Unavailable expected smoothness and residual for other samplings than b-nice")
    end

    stepsize = 0.0

    ## Checking the validity of the order of the step sizes
    if stochastic_stepsize > gradient_stepsize
        error("The gradient step size should be greater than the stochastic step size")
    elseif stochastic_stepsize < 0.0 || gradient_stepsize < 0.0
        error("Negative stochastic or gradient step size")
    end

    if 0 <= reference_update_proba <= 1
        reference_update_distrib = Bernoulli(reference_update_proba)
    else
        error("Invalid reference update probability")
    end

    reference_point = zeros(prob.numfeatures)
    reference_grad = zeros(prob.numfeatures)

    method = Leap_SVRG_method(epocsperiter, gradsperiter, number_computed_gradients, name, stepmethod, bootmethod, stepsize, stochastic_stepsize, gradient_stepsize, L, Lmax, expected_smoothness, expected_residual, reference_update_distrib, reference_point, reference_grad, reset, sampling)

    return method
end


"""
    boot_Leap_SVRG!(prob, method, options)

Modify the method to set the stepsize based on the smoothness constants of the problem stored in **Leap\\_SVRG\\_method** and possibly sets the number of skipped error calculation if not specfied such that 30 points are to be plotted.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **Leap\\_SVRG\\_method** method: Loopless-SVRG method created by `initiate_Leap_SVRG`
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
# OUTPUTS
- **NONE**
"""
function boot_Leap_SVRG!(prob::Prob, method::Leap_SVRG_method, options::MyOptions)
    if options.stepsize_multiplier > 0.0
        println("Manually set stochastic step size")
        if method.stochastic_stepsize == 0.0
            method.stochastic_stepsize = options.stepsize_multiplier
        end
        if method.gradient_stepsize == 0.0
            method.gradient_stepsize = 1/method.L
            println("Automatically set gradient step to 1/L")
        else
            println("Manually set gradient step size")
        end
    elseif options.stepsize_multiplier == -1.0
        if occursin("nice", method.sampling.name)
            method.stochastic_stepsize = 1/(2*(method.expected_smoothness + 2*method.expected_residual)) # theoretical optimal value for b-nice sampling
            method.gradient_stepsize = 1/method.L
            @printf "Automatically set Leap-SVRG theoretical stochastic step size: %1.3e and gradient step size: %1.3e\n" method.stochastic_stepsize method.gradient_stepsize
        else
            error("No theoretical step sizes available for Leap-SVRG with this sampling")
        end
    else
        error("Invalid options stepsize")
    end
    method.stepsize = method.gradient_stepsize # \alpha_0 = \eta

    # WARNING: The following if statement does not seem to modify the method that is returned afterwards...
    if options.skip_error_calculation == 0.0
        options.skip_error_calculation = ceil(options.max_epocs*prob.numdata/(options.batchsize*30)) # show 30 points between 0 and the max number of epochs
        # 20 points over options.max_epocs when there are options.max_epocs *prob.numdata/(options.batchsize)) iterates in total
    end
    println("Skipping ", options.skip_error_calculation, " iterations per epoch\n")
end


"""
    reset_Leap_SVRG(prob, method, options)

Reset the Loopless-SVRG, especially the step size, the point and gradient reference.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **Leap\\_SVRG\\_method** method: Loopless-SVRG method created by `initiate_Leap_SVRG`
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
# OUTPUTS
- **NONE**
"""
function reset_Leap_SVRG!(prob::Prob, method::Leap_SVRG_method, options::MyOptions)
    println("\n---- RESET LEAP-SVRG ----\n")

    method.number_computed_gradients = Int64[0]
    method.stepsize = 0.0 # Will be set during boot

    method.reference_point = zeros(prob.numfeatures)
    method.reference_grad = zeros(prob.numfeatures)
end