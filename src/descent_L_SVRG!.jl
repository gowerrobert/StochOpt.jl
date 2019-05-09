"""
    descent_L_SVRG!(x, prob, options, method, iter, d)

Compute the descent direction (d).

# INPUTS
- **Array{Float64}** x: point at the current iteration
- **Prob** prob: considered problem, i.e. logistic regression, ridge ression...
- **MyOptions** options: different options such as the mini-batch size, the stepsize_multiplier...
- **L\\_SVRG\\_method** method: Loopless-SVRG method created by `initiate_L_SVRG`
- **Int64** iter: current iteration
- **Array{Float64}** d: descent direction
# OUTPUTS
- **NONE**
"""
function descent_L_SVRG!(x::Array{Float64}, prob::Prob, options::MyOptions, method::L_SVRG_method, iter::Int64, d::Array{Float64})
    ## Initialization at first iteration
    if iter == 1
        println("Initialization of the reference point and gradient")
        method.reference_point[:] = x # initialize reference point

        if prob.numdata > 10000 || prob.numfeatures > 10000
            println("Dimensions are too large too compute the full gradient")
            s = sample(1:prob.numdata, 100, replace=false)
            method.reference_grad[:] = prob.g_eval(method.reference_point, s) # initialize stochastic reference gradient
        else
            method.reference_grad[:] = prob.g_eval(method.reference_point, 1:prob.numdata) # initialize reference gradient
        end
    end

    ## Sampling method
    s = method.sampling.sampleindices(method.sampling)
    # println("s: ", s)
    if isempty(s) # if no point is sampled
        d[:] = -method.reference_grad
    else
        d[:] = -prob.g_eval(x, s) + prob.g_eval(method.reference_point, s) - method.reference_grad
    end

    ## Stochastic update of the reference
    flipped_coin = rand(method.reference_update_distrib)
    if flipped_coin
        println("Update the reference point and gradient")
        method.reference_point[:] = x # update reference point

        if prob.numdata > 10000 || prob.numfeatures > 10000
            s = sample(1:prob.numdata, 100, replace=false)
            method.reference_grad[:] = prob.g_eval(method.reference_point, s) # update stochastic reference gradient
        else
            method.reference_grad[:] = prob.g_eval(method.reference_point, 1:prob.numdata) # update reference gradient
        end
    end

end
