"""
    descent_SVRG_vanilla!(x, prob, options, method, iter, d)

Compute the descent direction (d).

# INPUTS
- **Array{Float64}** x: point at the current iteration
- **Prob** prob: considered problem, i.e. logistic regression, ridge ression...
- **MyOptions** options: different options such as the mini-batch size, the stepsize_multiplier...
- **SVRG\\_vanilla\\_method** method: original SVRG method (option I) created by `initiate_SVRG_vanilla`
- **Int64** iter: current iteration
- **Array{Float64}** d: descent direction
# OUTPUTS
- **NONE**
"""
function descent_SVRG_vanilla!(x::Array{Float64}, prob::Prob, options::MyOptions, method::SVRG_vanilla_method, iter::Int64, d::Array{Float64})
    gradient_counter = 0 # number of stochastic gradients computed during this iteration

    ## SVRG outerloop
    if iter%method.numinneriters == 1 || method.numinneriters == 1 # reset reference point and gradient
        # println("SVRG outer loop at iteration: ", iter)
        method.reference_point[:] = x # option I: the reference is the last iterate

        if prob.numdata > 10^8 || prob.numfeatures > 10^8
            if iter == 1
                println("Dimensions are too large too compute the full gradient")
            end
            sampled_indices = sample(1:prob.numdata, 100, replace=false)
            method.reference_grad[:] = prob.g_eval(method.reference_point, sampled_indices) # reset a stochastic reference gradient
            gradient_counter += 100
        else
            method.reference_grad[:] = prob.g_eval(method.reference_point, 1:prob.numdata) # reset reference gradient
            gradient_counter += prob.numdata
        end

        d[:] = -method.reference_grad # the first iteration of the inner loop is equivalent to a gradient step because x = reference_point
    else
        ## SVRG inner step
        # println("        SVRG inner loop at iteration: ", iter)

        ## Sampling method
        sampled_indices = method.sampling.sampleindices(method.sampling)
        # println("sampled_indices: ", sampled_indices)
        if isempty(sampled_indices) # if no point is sampled
            d[:] = -method.reference_grad
        else
            d[:] = -prob.g_eval(x, sampled_indices) + prob.g_eval(method.reference_point, sampled_indices) - method.reference_grad
        end
        gradient_counter += 2*length(sampled_indices)
    end
    # println("|d| ", norm(d))

    ## Monitoring the number of computed gradient during this iteration
    method.number_computed_gradients = [method.number_computed_gradients method.number_computed_gradients[end] + gradient_counter]
end
