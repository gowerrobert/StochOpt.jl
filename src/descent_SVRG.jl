function descent_SVRG(x::Array{Float64}, prob::Prob, options::MyOptions, method::Method, N::Int64, d::Array{Float64})
    # SVRG outerloop
    if(N%method.numinneriters == 1 || method.numinneriters == 1)# Reset reference point, grad estimate and Hessian estimate
        # println("SVRG outer loop at iteration: ", N)
        method.prevx[:] = x;

        # method.grad[:] = prob.g_eval(x, 1:prob.numdata); # Out of memory error() when numdata or numfeatures are too large
        if prob.numdata > 10000 || prob.numfeatures > 10000
            if N==1 println("Dimensions are too large too compute the full gradient") end
            s = sample(1:prob.numdata, 100, replace=false);
            method.grad[:] = prob.g_eval(x, s); # Stochastic reference gradient
        else
            method.grad[:] = prob.g_eval(x, 1:prob.numdata); # Reference gradient
        end

        d[:] = -method.grad;
    else
        sample!(1:prob.numdata, method.ind, replace=false);
        #  s = sample!(1:prob.numdata, options.batchsize, replace=false);
        # SVRG inner step
        d[:] = -prob.g_eval(x, method.ind)+prob.g_eval(method.prevx, method.ind) - method.grad
    end
    #  println("|d| ", norm(d))
end
