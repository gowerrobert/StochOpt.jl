"""
    descent_free_SVRG_nice!(x::Array{Float64}, prob::Prob, options::MyOptions, method::free_SVRG_method, iter::Int64, d::Array{Float64})

Compute the descent direction (d)

# INPUTS
- **Array{Float64}** x: point at the current iteration
- **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)
- **MyOptions** options: different options such as the mini-batch size, the stepsize_multiplier... (see src/StochOpt.jl)
- **free\\_SVRG\\_method** method: Free-SVRG method created by `initiate_free_SVRG`
- **Int64** iter: current iteration
- **Array{Float64}** d: descent direction
# OUTPUTS
- **NONE**
"""
function descent_free_SVRG!(x::Array{Float64}, prob::Prob, options::MyOptions, method::free_SVRG_method, iter::Int64, d::Array{Float64})
    ## SVRG outerloop
    if iter%method.numinneriters == 1 || method.numinneriters == 1 # reset reference point and gradient
        # println("SVRG outer loop at iteration: ", iter)
        if isempty(method.averaging_weights)
            method.reference_point[:] = x; # Reference point set to last iterate iterates x^m
        else
            if iter == 1
                method.reference_point[:] = x; # Reference point set to initial point x_0^m
            else
                method.reference_point[:] = method.new_reference_point; # Reference point set to the weighted average of iterates from x^0 to x^{m-1}
            end
            # println("Resetting new_reference_point to zero")
            method.new_reference_point[:] = zeros(prob.numfeatures)
        end

        if prob.numdata > 10000 || prob.numfeatures > 10000
            if iter == 1
                println("Dimensions are too large too compute the full gradient")
            end
            sampled_indices = sample(1:prob.numdata, 100, replace=false);
            method.reference_grad[:] = prob.g_eval(method.reference_point, sampled_indices) # reset a stochastic reference gradient
            method.number_computed_gradients += 100
        else
            method.reference_grad[:] = prob.g_eval(method.reference_point, 1:prob.numdata) # reset reference gradient
            method.number_computed_gradients += prob.numdata
        end

        # d[:] = -method.reference_grad; # WRONG: the first iteration of the inner loop is equivalent to a gradient step
    end
    ## SVRG inner step
    # println("---- SVRG inner loop at iteration: ", iter)
    if !isempty(method.averaging_weights)
        if iter % method.numinneriters == 0 # small index shift: weight a_m times point x_{s+1}^{m-1}
            idx_weights = method.numinneriters; # i = m
        else
            idx_weights = iter % method.numinneriters; # for i = 1, ..., m-1
        end
        # println("        idx weights: ", idx_weights)
        method.new_reference_point[:] += method.averaging_weights[idx_weights] .* x;
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

    #  println("|d| ", norm(d))
end
