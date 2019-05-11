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
            sampled_indices = sample(1:prob.numdata, 100, replace=false)
            method.reference_grad[:] = prob.g_eval(method.reference_point, sampled_indices) # initialize stochastic reference gradient
            method.number_computed_gradients += 100
        else
            method.reference_grad[:] = prob.g_eval(method.reference_point, 1:prob.numdata) # initialize reference gradient
            method.number_computed_gradients += prob.numdata
        end
    end

    ## Sampling method
    sampled_indices = method.sampling.sampleindices(method.sampling)
    # println("sampled_indices: ", sampled_indices)
    if isempty(sampled_indices) # if no point is sampled
        d[:] = -method.reference_grad
    else
        d[:] = -prob.g_eval(x, sampled_indices) + prob.g_eval(method.reference_point, sampled_indices) - method.reference_grad
    end
    method.number_computed_gradients += 2*length(sampled_indices)

    ## Stochastic update of the reference
    flipped_coin = rand(method.reference_update_distrib)
    if flipped_coin
        println("Update the reference point and gradient")
        method.reference_point[:] = x # update reference point

        if prob.numdata > 10000 || prob.numfeatures > 10000
            sampled_indices = sample(1:prob.numdata, 100, replace=false)
            method.reference_grad[:] = prob.g_eval(method.reference_point, sampled_indices) # update stochastic reference gradient
            method.number_computed_gradients += 100
        else
            method.reference_grad[:] = prob.g_eval(method.reference_point, 1:prob.numdata) # update reference gradient
            method.number_computed_gradients += prob.numdata
        end
    end

end
