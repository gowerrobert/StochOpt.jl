function boot_2D(prob::Prob, method::Method,options::MyOptions)
    method.H = zeros(length(method.prevx));#prob.Hess_D(method.prevx, 1:prob.numdata);
    method.S = zeros(length(method.prevx));
    method.gradsperiter = 3*options.batchsize + 2*prob.numdata/method.numinneriters + 1; #includes the cost of computing the diagonal Hessian approximation.
    method.name = string("2D");#-",options.batchsize);
    method.stepmethod = descent_2D;
    if(options.precondition)
        method.name = string(method.name, "-qN");
    end
    return method;
end