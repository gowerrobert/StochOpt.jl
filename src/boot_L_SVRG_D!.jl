"""
    initiate_L_SVRG_D(prob, options ; numinneriters=0, probs=[])

Initiate the Loopless-SVRG-Decreasing method.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
- **Sampling** sampling: sampling object (b-nice or independent sampling)
- **Float64** reference_update_proba: probability of updating the reference at each iteration of the (inner and only) loop (possible value: p = 1/n)
# OUTPUTS
- **L\\_SVRG\\_D\\_method** method: Loopless-SVRG-Decreasing method created by `initiate_L_SVRG_D`

# REFERENCES
__Our Title__\\
Francis Bach, Othmane Sebbouh, Robert M. Gower and Nidham Gazagnadou\\
arXiv:??????, 2019
"""
function initiate_L_SVRG_D(prob::Prob, options::MyOptions, sampling::Sampling, reference_update_proba::Float64)
    n = prob.numdata

    ## No deterministic number of computed gradients per iteration because of the inner and outer loop scheme
    b = sampling.batchsize # deterministic or average mini-batch size
    epocsperiter = 0
    gradsperiter = 0
    number_computed_gradients = Int64[0] # dynamic table of the number of computed gradients at each iteration

    name = string("L-SVRG-D-", sampling.name)

    stepmethod = descent_L_SVRG_D!
    bootmethod = boot_L_SVRG_D!
    reset = reset_L_SVRG_D!

    stepsize = 0.0
    initial_stepsize = 0.0

    L = prob.L
    Lmax = prob.Lmax

    if occursin("nice", sampling.name)
        expected_smoothness = ((n-b)/(b*(n-1)))*Lmax + ((n*(b-1))/(b*(n-1)))*L
        expected_residual = ((n-b)/(b*(n-1)))*Lmax
    else
        error("Unavailable expected smoothness and residual for other samplings than b-nice")
    end
    # if numinneriters == -1
    #     error("No theoretical inner loop size available for L-SVRG-D with this sampling") # no theoretical value given by Kovalev et al
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

    method = L_SVRG_D_method(epocsperiter, gradsperiter, number_computed_gradients, name, stepmethod, bootmethod, stepsize, initial_stepsize, L, Lmax, expected_smoothness, reference_update_proba, reference_update_distrib, reference_point, reference_grad, reset, sampling)

    # epocsperiter::Float64
    # gradsperiter::Float64
    # number_computed_gradients::Array{Int64} # cumulative sum of the number of computed stochastic gradients at each iteration
    # name::AbstractString
    # stepmethod::Function # /!\ mutating function
    # bootmethod::Function # /!\ mutating function
    # stepsize::Float64 # step size
    # initial_stepsize::Float64 # step size at first iteration
    # L::Float64 # smoothness constant of the whole objective function f
    # Lmax::Float64 # max of the smoothness constant of the f_i functions
    # expected_smoothness::Float64 # Expected smoothness constant
    # reference_update_proba::Float64 # probability of updating the reference point and gradient (denoted p in the paper)
    # reference_update_distrib::Bernoulli{Float64} # Bernoulli distribution controlling the frequence of update of the reference point and gradient
    # reference_point::Array{Float64}
    # reference_grad::Array{Float64}
    # reset::Function # reset some parameters of the method
    # sampling::Sampling # b-nice or independent sampling


    return method
end


"""
    boot_L_SVRG_D!(prob, method, options)

Modify the method to set the initial stepsize based on the smoothness constants of the problem stored in **L\\_SVRG\\_D\\_method** and possibly sets the number of skipped error calculation if not specfied such that 30 points are to be plotted.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **L\\_SVRG\\_D\\_method** method: Loopless-SVRG-Decreasing method created by `initiate_L_SVRG_D`
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
# OUTPUTS
- **NONE**
"""
function boot_L_SVRG_D!(prob::Prob, method::L_SVRG_D_method, options::MyOptions)
    if options.stepsize_multiplier > 0.0
        println("Manually set initial step size")
        method.stepsize = options.stepsize_multiplier
    elseif options.stepsize_multiplier == -1.0
        if occursin("nice", method.sampling.name)
            p = method.reference_update_proba
            aux = ( p*(2-p)*(3-2*p) ) / ( (7-4*p)*(1-(1-p)^1.5) )
            method.initial_stepsize =  aux / (2 * method.expected_smoothness)   # theoretical value
            method.stepsize = method.initial_stepsize
            println("Automatically set L-SVRG-D initial step size: ", method.stepsize)
        else
            error("No theoretical step size available for L-SVRG-D with this sampling")
        end
        println("Theoretical initial step size: ", method.stepsize)
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
    reset_L_SVRG_D(prob, method, options)

Reset the Loopless-SVRG-Decreasing, especially the step size, the point and gradient reference.

# INPUTS
- **Prob** prob: considered problem, e.g., logistic regression, ridge regression...
- **L\\_SVRG\\_D\\_method** method: Loopless-SVRG-Decreasing method created by `initiate_L_SVRG_D`
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
# OUTPUTS
- **NONE**
"""
function reset_L_SVRG_D!(prob::Prob, method::L_SVRG_D_method, options::MyOptions)
    println("\n---- RESET LOOPLESS-SVRG-DECREASING ----\n")

    method.number_computed_gradients = Int64[0]
    method.stepsize = 0.0          # Will be set during boot
    method.initial_stepsize = 0.0 # Will be set during boot

    method.reference_point = zeros(prob.numfeatures)
    method.reference_grad = zeros(prob.numfeatures)
end