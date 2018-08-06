function descent_grad(x::Array{Float64}, prob::Prob, options::MyOptions, method::Method, i::Int64, d::Array{Float64})
    s = sample(1:prob.numdata, options.batchsize, replace=false);
    d[:] = -prob.g_eval(x, s);
    # Saga
    #prob.HJ = zeros(prob.numfeatures, prob.numdata);
end
