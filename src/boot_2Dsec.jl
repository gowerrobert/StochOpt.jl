function boot_2Dsec(prob::Prob, method::Method, options::MyOptions)
    method.S = zeros(prob.numfeatures);
    method.Sold = zeros(prob.numfeatures);
    method.H = zeros(prob.numfeatures);
    method.diffpnt = zeros(prob.numfeatures);
    method.gradsperiter = 4*options.batchsize + 4*prob.numdata/method.numinneriters + 1.0; #includes the cost of performing the Hessian vector product.
    if(options.aux == Inf) options.aux = 0.01; end
    method.name = string("2Dsec");#-sig-2^",log(2,options.aux));
    method.stepmethod = descent_2Dsec;
    if(options.precondition)
       method.name = string(method.name, "-qN");
    end
    return method
end