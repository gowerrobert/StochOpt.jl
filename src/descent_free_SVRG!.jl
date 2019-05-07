"""
    descent_free_SVRG_nice!(x::Array{Float64}, prob::Prob, options::MyOptions, method::free_SVRG_nice_method, iter::Int64, d::Array{Float64})

Compute the descent direction (d)

# INPUTS
- **Array{Float64}** x: point at the current iteration
- **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)
- **MyOptions** options: different options such as the mini-batch size, the stepsize_multiplier... (see src/StochOpt.jl)
- **free\\_SVRG\\_nice\\_method** method: method of Free-SVRG for b-nice sampling
- **Int64** iter: current iteration
- **Array{Float64}** d: descent direction
# OUTPUTS
- **NONE**
"""
function descent_free_SVRG_nice!(x::Array{Float64}, prob::Prob, options::MyOptions, method::free_SVRG_nice_method, iter::Int64, d::Array{Float64})
    ## SVRG outerloop
    if iter%method.numinneriters == 1 || method.numinneriters == 1 # reset reference point and gradient
        println("SVRG outer loop at iteration: ", iter)
        if isempty(method.averaging_weights)
            method.reference_point[:] = x; # Reference point set to last iterate iterates x^m
        else
            if iter == 1
                method.reference_point[:] = x; # Reference point set to initial point x_0^m
            else
                method.reference_point[:] = method.new_reference_point; # Reference point set to the average of iterates from x^0 to x^{m-1}
            end
            println("Resetting new_reference_point to zero")
            println("        idx weights: 1")
            method.new_reference_point[:] = method.averaging_weights[1] .* x;
        end

        if prob.numdata > 10000 || prob.numfeatures > 10000
            if iter==1
                println("Dimensions are too large too compute the full gradient")
            end
            s = sample(1:prob.numdata, 100, replace=false);
            method.reference_grad[:] = prob.g_eval(x, s); # reset a stochastic reference gradient
        else
            method.reference_grad[:] = prob.g_eval(x, 1:prob.numdata); # reset reference gradient
        end

        d[:] = -method.reference_grad; # the first iteratation of the inner loop is equivalent to a gradient step
    else
        ## SVRG inner step
        println("---- SVRG inner loop at iteration: ", iter)
        if !isempty(method.averaging_weights)
            if iter % method.numinneriters == 0 # small index shift
                idx_weights = method.numinneriters;
            else
                idx_weights = iter % method.numinneriters;
            end
            println("        idx weights: ", idx_weights)
            method.new_reference_point[:] += method.averaging_weights[idx_weights] .* x;
        end
        s = sample(1:prob.numdata, options.batchsize, replace=false); # b-nice sampling
        # s = independent_sampling(method.probs) # independent_sampling
        d[:] = -prob.g_eval(x, s) + prob.g_eval(method.reference_point, s) - method.reference_grad
    end


    # ## SVRG outerloop
    # if iter%method.numinneriters == 1 || method.numinneriters == 1 # reset reference point and gradient
    #     # println("SVRG outer loop at iteration: ", iter)
    #     if isempty(method.averaging_weights) || iter == 1
    #         method.reference_point[:] = x; # Reference point set to last iterate iterates x^m
    #     else
    #         method.reference_point[:] = method.new_reference_point; # Reference point set to the average of iterates from x^0 to x^{m-1}
    #         # println("Resetting new_reference_point to zero")
    #         method.new_reference_point = zeros(prob.numfeatures);
    #     end

    #     if prob.numdata > 10000 || prob.numfeatures > 10000
    #         if iter==1
    #             println("Dimensions are too large too compute the full gradient")
    #         end
    #         s = sample(1:prob.numdata, 100, replace=false);
    #         method.reference_grad[:] = prob.g_eval(method.reference_point, s); # Reset a stochastic reference gradient
    #     else
    #         method.reference_grad[:] = prob.g_eval(method.reference_point, 1:prob.numdata); # Reset reference gradient
    #     end
    # end

    # if !isempty(method.averaging_weights)
    #     # println("--------- Current iter: ", iter);
    #     if iter % method.numinneriters == 0
    #         idx_weights = method.numinneriters;
    #     else
    #         idx_weights = iter % method.numinneriters;
    #     end
    #     # println("------------------ idx weights: ", idx_weights)
    #     method.new_reference_point[:] += method.averaging_weights[idx_weights] .* x;
    # end

    # ## SVRG inner step
    # # println("        SVRG inner loop at iteration: ", iter)
    # s = sample(1:prob.numdata, options.batchsize, replace=false);
    # d[:] = -prob.g_eval(x, s) + prob.g_eval(method.reference_point, s) - method.reference_grad
    # #  println("|d| ", norm(d))
end
