function boot_SVRG(prob::Prob, method::Method, options::MyOptions)
    method.name = string("SVRG")#,options.batchsize);
    method.stepmethod = descent_SVRG;
    return method;
end
