function boot_SVRG2(prob::Prob, method::Method, options::MyOptions)
    #method.H = prob.Hess_eval(method.prevx, 1:prob.numdata);
    method.Hsp = spzeros(prob.numfeatures, prob.numfeatures);
    method.diffpnt = zeros(length(method.prevx));
    method.gradsperiter = 3*options.batchsize + 2*prob.numdata/method.numinneriters + prob.numfeatures;
    #gradsperiter = 2*options.batchsize + prob.numdata/numinneriters;
    #includes the cost of performing the Hessian vector product.
    method.name = "SVRG2"; #string("SVRG2-", options.batchsize);
    method.stepmethod = descent_SVRG2;
    return method;
end
