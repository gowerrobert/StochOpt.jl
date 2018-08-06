function descent_SVRG2(x::Array{Float64}, prob::Prob, options::MyOptions, method::Method, N::Int64, d::Array{Float64})
    # SVRG2, a second order variant of SVRG that uses the full Hessian
    if(N%method.numinneriters == 1) # Reset reference point, grad estimate and Hessian estimate
        #  println("SVRG2 outer loop at iteration: ", N, "  sum(H): ", sum(method.H))
        method.prevx[:] = x;
        method.grad[:] = prob.g_eval(x, 1:prob.numdata);
        prob.Hess_eval!(x, 1:prob.numdata, method.grad, method.Hsp);
        d[:] = -method.grad;
    else     # SVRG2 inner step
        sample!(1:prob.numdata, method.ind, replace=false);
        method.diffpnt[:] = x-method.prevx; #diffpnt = x-method.prevx;
        d[:] = -prob.g_eval(x,method.ind) + prob.g_eval(method.prevx, method.ind) + prob.Hess_opt(method.prevx, method.ind, method.diffpnt) - method.grad - method.Hsp*method.diffpnt;
    end
end