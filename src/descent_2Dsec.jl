function descent_2Dsec(x::Array{Float64}, prob::Prob, options::MyOptions, method::Method, N::Int64, d::Array{Float64})
    # Calculate a descent direction using a Hessian approximation built by using the secant equation combined with a diagonal approximation
    sig=options.aux; # unfortunately using the name restol to hold the sigma value
    # SVRG outerloop
    if(N%method.numinneriters == 1)# Reset reference point, grad estimate and Hessian estimate
        #println("sec outer  iteration: ",N)
        method.diffpnt[:] = x - method.prevx;
        method.H[:] = (prob.Hess_opt(x, 1:prob.numdata, method.diffpnt).*method.diffpnt +
            sig.*prob.Hess_D(x, 1:prob.numdata))./(method.diffpnt.*method.diffpnt.+sig); #(gradnew-method.grad)./(x- method.prevx);
        method.grad[:] = prob.g_eval(x, 1:prob.numdata);
        d[:] = -method.grad;
        method.Sold[:] = method.prevx; ## A place holder I'm using to store the prevprevx;
        method.prevx[:] = x;
    elseif(N > method.numinneriters)   # SVRG2 inner step  println("inner loop")
        sample!(1:prob.numdata, method.ind, replace=false);
        method.diffpnt[:] = x - method.prevx;
        #ppx = method.J[:, 1];
        d[:] = -prob.g_eval(x, method.ind) + prob.g_eval(method.prevx, method.ind) +
            (prob.Hess_opt(method.prevx, method.ind, method.prevx - method.Sold[:]).*(method.prevx - method.Sold[:]) + 
            sig.*prob.Hess_D(method.prevx, method.ind))./ ((method.prevx - method.Sold[:]).*(method.prevx - method.Sold[:]).+sig).*method.diffpnt
            - method.grad - method.H.*method.diffpnt; #
    else # SVRG  println("plain SVRG")
        sample!(1:prob.numdata, method.ind, replace=false);
        d[:] = -prob.g_eval(x, method.ind) + prob.g_eval(method.prevx, method.ind) - method.grad;
    end
    if(options.precondition && N > method.numinneriters) #println("qN")
        d[:] = (method.H).^(-1).*d;
    end
end
