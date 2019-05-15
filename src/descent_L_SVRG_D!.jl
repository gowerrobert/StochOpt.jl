"""
    descent_L_SVRG_D!(x, prob, options, method, iter, d)

Compute the descent direction (d).

# INPUTS
- **Array{Float64}** x: point at the current iteration
- **Prob** prob: considered problem, i.e. logistic regression, ridge ression...
- **MyOptions** options: different options such as the mini-batch size, the stepsize_multiplier...
- **L\\_SVRG\\_D\\_method** method: Loopless-SVRG-Decreasing method created by `initiate_L_SVRG_D`
- **Int64** iter: current iteration
- **Array{Float64}** d: descent direction
# OUTPUTS
- **NONE**
"""
function descent_L_SVRG_D!(x::Array{Float64}, prob::Prob, options::MyOptions, method::L_SVRG_D_method, iter::Int64, d::Array{Float64})
    gradient_counter = 0 # number of stochastic gradients computed during this iteration

    ## Initialization at first iteration
    if iter == 1
        println("Initialization of the reference point and gradient")
        method.reference_point[:] = x # initialize reference point

        if prob.numdata > 10^8 || prob.numfeatures > 10^8
            println("Dimensions are too large too compute the full gradient")
            sampled_indices = sample(1:prob.numdata, 100, replace=false)
            method.reference_grad[:] = prob.g_eval(method.reference_point, sampled_indices) # initialize stochastic reference gradient
            gradient_counter += 100
        else
            method.reference_grad[:] = prob.g_eval(method.reference_point, 1:prob.numdata) # initialize reference gradient
            gradient_counter += prob.numdata
        end
    end

    if norm(x - method.reference_point) < 1e-7
        println("iter: ", iter, ", x = ref point")
    end

    ## Sampling method
    sampled_indices = method.sampling.sampleindices(method.sampling)
    # println("sampled_indices: ", sampled_indices)
    if isempty(sampled_indices) # if no point is sampled
        d[:] = -method.reference_grad
    else
        d[:] = -prob.g_eval(x, sampled_indices) + prob.g_eval(method.reference_point, sampled_indices) - method.reference_grad
    end
    gradient_counter += 2*length(sampled_indices)

    ## Stochastic update of the reference
    flipped_coin = rand(method.reference_update_distrib)
    if flipped_coin
        println("Update the reference point and gradient")
        method.reference_point[:] = x # update reference point

        if prob.numdata > 10^8 || prob.numfeatures > 10^8
            sampled_indices = sample(1:prob.numdata, 100, replace=false)
            method.reference_grad[:] = prob.g_eval(method.reference_point, sampled_indices) # update stochastic reference gradient
            gradient_counter += 100
        else
            method.reference_grad[:] = prob.g_eval(method.reference_point, 1:prob.numdata) # update reference gradient
            gradient_counter += prob.numdata
        end

        ## Decrease the step size
        method.stepsize *= sqrt(1-method.reference_update_proba)
    end

    ## Monitoring the number of computed gradient during this iteration
    method.number_computed_gradients = [method.number_computed_gradients method.number_computed_gradients[end] + gradient_counter]
end
