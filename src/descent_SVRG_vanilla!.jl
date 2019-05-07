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
    ## SVRG outerloop
    if iter%method.numinneriters == 1 || method.numinneriters == 1 # reset reference point and gradient
        # println("SVRG outer loop at iteration: ", iter)
        method.reference_point[:] = x; # option I: the reference is the last iterate

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
        # println("        SVRG inner loop at iteration: ", iter)

        ## Sampling method
        if method.sampling == "nice"
            s = sample(1:prob.numdata, options.batchsize, replace=false); # b-nice sampling
            # println("\n\nSAMPLING: type of s: ", typeof(s))
        elseif method.sampling == "independent"
            s = independent_sampling(method.probs) # independent_sampling
        end

        d[:] = -prob.g_eval(x, s) + prob.g_eval(method.reference_point, s) - method.reference_grad
    end

    # println("|d| ", norm(d))
end
