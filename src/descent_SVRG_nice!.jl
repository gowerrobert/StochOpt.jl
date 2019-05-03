"""
    descent_SVRG_nice!(x::Array{Float64}, prob::Prob, options::MyOptions, method::SVRG_nice_method, iter::Int64, d::Array{Float64})

Compute the descent direction (d)

#INPUTS:\\
    - **Array{Float64}** x: point at the current iteration
    - **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)\\
    - **MyOptions** options: different options such as the mini-batch size, the stepsize_multiplier... (see src/StochOpt.jl)\\
    - **SVRG_nice_method** method: method of SVRG for b-nice sampling\\
    - **Int64** iter: current iteration\\
    - **Array{Float64}** d: descent direction\\
#OUTPUTS:\\
    - NONE
"""
function descent_SVRG_nice!(x::Array{Float64}, prob::Prob, options::MyOptions, method::SVRG_nice_method, iter::Int64, d::Array{Float64})
    # SVRG outerloop
    if iter%method.numinneriters == 1 || method.numinneriters == 1 # Reset reference point, grad estimate and Hessian estimate
        println("SVRG outer loop at iteration: ", iter)
        method.reference_point[:] = x;

        if prob.numdata > 10000 || prob.numfeatures > 10000
            if iter==1
                println("Dimensions are too large too compute the full gradient")
            end
            s = sample(1:prob.numdata, 100, replace=false);
            method.reference_grad[:] = prob.g_eval(x, s); # Stochastic reference gradient
        else
            method.reference_grad[:] = prob.g_eval(x, 1:prob.numdata); # Reference gradient
        end

        d[:] = -method.reference_grad;
    else
        # SVRG inner step
        s = sample(1:prob.numdata, options.batchsize, replace=false);
        d[:] = -prob.g_eval(x, s) + prob.g_eval(method.reference_point, s) - method.reference_grad
    end
    #  println("|d| ", norm(d))
end
