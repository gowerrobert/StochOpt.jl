"""
    descent_Leap_SVRG!(x, prob, options, method, iter, d)

Compute the descent direction (d).

# INPUTS
- **Array{Float64}** x: point at the current iteration
- **Prob** prob: considered problem, i.e. logistic regression, ridge ression...
- **MyOptions** options: different options such as the mini-batch size, the stepsize_multiplier...
- **Leap\\_SVRG\\_method** method: Loopless-SVRG method created by `initiate_Leap_SVRG`
- **Int64** iter: current iteration
- **Array{Float64}** d: descent direction
# OUTPUTS
- **NONE**
"""
function descent_Leap_SVRG!(x::Array{Float64}, prob::Prob, options::MyOptions, method::Leap_SVRG_method, iter::Int64, d::Array{Float64})
    gradient_counter = 0 # number of stochastic gradients computed during this iteration

    # println("ref point norm: ", norm(method.reference_point))

    if iter > 1
        flipped_coin = rand(method.reference_update_distrib)
    end

    ## Initialization
    if iter == 1
        println("Initialization of the reference point and gradient")
        method.reference_point[:] = x # initialize reference point

        # update reference gradient: mu = \nabla f(w^0)
        if prob.numdata > 10^8 || prob.numfeatures > 10^8
            println("Dimensions are too large too compute the full gradient")
            sampled_indices = sample(1:prob.numdata, 100, replace=false)
            gradient_counter += 100
        else
            method.reference_grad[:] = prob.g_eval(method.reference_point, 1:prob.numdata) # initialize full reference gradient
            gradient_counter += prob.numdata
        end

        # method.stepsize = method.gradient_stepsize # \alpha_0 = \eta, should already be set in boot
    ## Stochastic update of the reference and the step size
    elseif flipped_coin
        # println("Update the reference point, gradient and step size")
        method.reference_point[:] = x # update reference point: w^{k+1} = x^{k+1}

        # update reference gradient: mu = \nabla f(w^{k+1})
        if prob.numdata > 10^8 || prob.numfeatures > 10^8
            sampled_indices = sample(1:prob.numdata, 100, replace=false)
            method.reference_grad[:] = prob.g_eval(method.reference_point, sampled_indices) # update stochastic reference gradient
            gradient_counter += 100
        else
            method.reference_grad[:] = prob.g_eval(method.reference_point, 1:prob.numdata) # update full reference gradient
            gradient_counter += prob.numdata
        end

        method.stepsize = method.gradient_stepsize # \alpha_{k+1} = \eta
    else
        # println("Update only the step size")
        method.stepsize = method.stochastic_stepsize # \alpha_{k+1} = \alpha
    end

    ## Sampling method
    sampled_indices = method.sampling.sampleindices(method.sampling)
    # println("sampled_indices: ", sampled_indices)
    if isempty(sampled_indices) # if no point is sampled
        d[:] = -method.reference_grad
    else
        if norm(x - method.reference_point) < 1e-7
            println("x = ref point")
        end
        d[:] = -prob.g_eval(x, sampled_indices) + prob.g_eval(method.reference_point, sampled_indices) - method.reference_grad
    end
    gradient_counter += 2*length(sampled_indices)

    ## Monitoring the number of computed gradient during this iteration
    method.number_computed_gradients = [method.number_computed_gradients method.number_computed_gradients[end] + gradient_counter]
end
