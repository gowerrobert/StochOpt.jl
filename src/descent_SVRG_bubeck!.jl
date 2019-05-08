"""
    descent_SVRG_bubeck!(x, prob, options, method, iter, d)

Compute the descent direction (d).

# INPUTS
- **Array{Float64}** x: point at the current iteration
- **Prob** prob: considered problem, i.e. logistic regression, ridge ression...
- **MyOptions** options: different options such as the mini-batch size, the stepsize_multiplier...
- **SVRG\\_bubeck\\_method** method: original SVRG method (option I) created by `initiate_SVRG_bubeck`
- **Int64** iter: current iteration
- **Array{Float64}** d: descent direction
# OUTPUTS
- **NONE**
"""
function descent_SVRG_bubeck!(x::Array{Float64}, prob::Prob, options::MyOptions, method::SVRG_bubeck_method, iter::Int64, d::Array{Float64})
    ## SVRG outerloop
    if iter%method.numinneriters == 1 || method.numinneriters == 1 # reset reference point and gradient
        # println("SVRG outer loop at iteration: ", iter)
        if iter == 1
            method.reference_point[:] = x; # Reference point set to initial point x_0^m
        else
            method.reference_point[:] = method.new_reference_point; # Reference point set to the average of iterates from x^0 to x^{m-1}
        end

        if prob.numdata > 10000 || prob.numfeatures > 10000
            if iter==1
                println("Dimensions are too large too compute the full gradient")
            end
            s = sample(1:prob.numdata, 100, replace=false);
            method.reference_grad[:] = prob.g_eval(method.reference_point, s); # reset a stochastic reference gradient
        else
            method.reference_grad[:] = prob.g_eval(method.reference_point, 1:prob.numdata); # reset reference gradient
        end

        d[:] = -method.reference_grad; # WRONG: the first iteratation of the inner loop is equivalent to a gradient step
    else
        ## SVRG inner step
        # println("        SVRG inner loop at iteration: ", iter)

        ## Sampling method
        s = method.sampling.sampleindices(method.sampling)
        # println("s: ", s)
        if isempty(s) # if no point is sampled
            d[:] = -method.reference_grad
        else
            d[:] = -prob.g_eval(x, s) + prob.g_eval(method.reference_point, s) - method.reference_grad
        end
    end

    # println("|d| ", norm(d))
end
