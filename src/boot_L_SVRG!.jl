"""
    initiate_L_SVRG(prob, options ; numinneriters=0, probs=[])

Initiate the Loopless-SVRG method.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
- **Sampling** sampling: sampling object (b-nice or independent sampling)
- **Float64** reference_update_proba: probability of updating the reference at each iteration of the (inner and only) loop (possible value: p = 1/n)
# OUTPUTS
- **L\\_SVRG\\_method** method: Loopless-SVRG method created by `initiate_L_SVRG`

# REFERENCES
__Don't Jump Through Hoops and Remove Those Loops: SVRG and Katyusha are Better Without the Outer Loop__\\
Dmitry Kovalev, Samuel Horvath and Peter Richtarik\\
arXiv:1901.08689, 2019.
"""
function initiate_L_SVRG(prob::Prob, options::MyOptions, sampling::Sampling, reference_update_proba::Float64)
    n = prob.numdata
    b = sampling.batchsize # sampling instead of options
    epocsperiter = b/n
    gradsperiter = b

    name = string("L-SVRG-", sampling.name)

    stepmethod = descent_L_SVRG!
    bootmethod = boot_L_SVRG!
    reset = reset_L_SVRG!

    stepsize = 0.0

    Lmax = prob.Lmax

    # if numinneriters == -1
    #     error("No theoretical inner loop size available for L-SVRG with this sampling") # no theoretical value given by Kovalev et al
    # elseif numinneriters < -1 || numinneriters == 0
    #     error("Invalid inner loop size")
    # end

    if 0 <= reference_update_proba <= 1
        reference_update_distrib = Bernoulli(reference_update_proba)
    else
        error("Invalid reference update probability")
    end

    reference_point = zeros(prob.numfeatures)
    reference_grad = zeros(prob.numfeatures)

    method = L_SVRG_method(epocsperiter, gradsperiter, name, stepmethod, bootmethod, stepsize, Lmax, reference_update_distrib, reference_point, reference_grad, reset, sampling)

    return method
end


"""
    boot_L_SVRG!(prob, method, options)

Modify the method to set the stepsize based on the smoothness constants of the problem stored in **L\\_SVRG\\_method** and possibly sets the number of skipped error calculation if not specfied such that 30 points are to be plotted.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **L\\_SVRG\\_method** method: Loopless-SVRG method created by `initiate_L_SVRG`
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
# OUTPUTS
- **NONE**
"""
function boot_L_SVRG!(prob::Prob, method::L_SVRG_method, options::MyOptions)
    if options.stepsize_multiplier > 0.0
        println("Manually set step size")
        method.stepsize = options.stepsize_multiplier
    elseif options.stepsize_multiplier == -1.0
        if method.sampling.name == "nice"
            method.stepsize = 1/(6*method.Lmax) # theoretical optimal value for 1-nice sampling by Kovalev et al
            println("Automatically set L-SVRG step size: ", method.stepsize)
        else
            error("No theoretical step size available for L-SVRG with this sampling")
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
    reset_L_SVRG(prob, method, options)

Reset the Loopless-SVRG, especially the step size, the point and gradient reference.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **L\\_SVRG\\_method** method: Loopless-SVRG method created by `initiate_L_SVRG`
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
# OUTPUTS
- **NONE**
"""
function reset_L_SVRG!(prob::Prob, method::L_SVRG_method, options::MyOptions)
    println("\n---- RESET LOOPLESS-SVRG ----\n")

    method.stepsize = 0.0 # Will be set during boot

    method.reference_point = zeros(prob.numfeatures)
    method.reference_grad = zeros(prob.numfeatures)
end