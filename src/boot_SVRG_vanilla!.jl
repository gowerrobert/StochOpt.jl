"""
    initiate_SVRG_vanilla(prob, options ; numinneriters=0, probs=[])

Initiate the original SVRG method with option I, for b-nice or independent sampling.
It uniformly picks b data points out of n at each iteration to build an estimate of the gradient.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
- **AbstractString** sampling: type of sampling b-nice (if set to "nice") or independent (if set to "independent")
- **Int64** numinneriters: size of the inner loop (twice the number of data samples n if set to 0)
- **Array{Float64}** probs: probability of selecting each coordinate (used for independent sampling)
# OUTPUTS
- **SVRG\\_vanilla\\_method** method: original SVRG method (option I) created by `initiate_SVRG_vanilla`

# REFERENCES
__Accelerating stochastic gradient descent using predictive variance reduction__\\
Rie Johnson and Tong Zhang\\
Advances in neural information processing systems. 2013.
"""
function initiate_SVRG_vanilla(prob::Prob, options::MyOptions, sampling::AbstractString ; numinneriters::Int64=0, probs::Array{Float64}=Float64[])
    n = prob.numdata
    b = options.batchsize
    epocsperiter = b/n
    gradsperiter = b

    name = "SVRG-vanilla"
    if sampling == "independent" # independent sampling
        if isempty(probs) || length(probs) != n
            error("Uncorrect probabilities")
        else
            if all(y->y==probs[1], probs) ## check if the probabilities are uniform
                avg_cardinal = round(Int64, sum(probs)) ## estimate of the average cardinal of the mini-batch
                name = string(name, "-", avg_cardinal)
            end
            name = string(name, "-indep")
        end
    elseif sampling == "nice" # b-nice sampling
        if b > 1
            name = string(name, "-", b)
        end
    else
        error("Unknown sampling procedure")
    end
    name = string(name, "-", sampling)

    stepmethod = descent_SVRG_vanilla!
    bootmethod = boot_SVRG_vanilla!
    reset = reset_SVRG_vanilla!

    stepsize = 0.0
    probs = []

    Lmax = prob.Lmax
    mu = prob.mu

    if numinneriters == 0
        numinneriters = 2*n
    end
    reference_point = zeros(prob.numfeatures)
    reference_grad = zeros(prob.numfeatures)

    method = SVRG_vanilla_method(epocsperiter, gradsperiter, name, stepmethod, bootmethod, b, stepsize, probs, Lmax, mu, numinneriters, reference_point, reference_grad, reset, sampling)

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

    method.batchsize = options.batchsize
    method.stepsize = options.stepsize_multiplier

    method.reference_point = zeros(prob.numfeatures)
    method.reference_grad = zeros(prob.numfeatures)
end