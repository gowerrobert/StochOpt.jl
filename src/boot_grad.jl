function boot_grad(prob::Prob, method::Method, options::MyOptions)
    # Lmean = mean(sum(prob.X.^2,1))+prob.lambda
    # method.stepsize = 0.25/Lmean;
    #method = Method(epocsperiter,"grad",descent_grad, [0],[0],[0.0], stepsize,[0],1);
    method.name = "grad";
    method.epocsperiter = options.batchsize/prob.numdata;
    method.stepmethod = descent_grad;
    return method;
    #prob.HJ = zeros(prob.numfeatures, prob.numdata);
end