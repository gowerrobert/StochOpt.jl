function boot_SPIN(prob::Prob,method, options::MyOptions)
    method.stepsize = options.stepsize_multiplier;
    if(options.skip_error_calculation == 0.0)
        options.skip_error_calculation = 10; # skip 10
    end
    println("Skipping ", options.skip_error_calculation, " iterations per epoch")   
    # prob.g_eval!(x, 1:prob.numdata, method.dn)
    return method
end

function initiate_SPIN(prob::Prob, options::MyOptions; sketchsize=convert(Int64, floor(sqrt(prob.numfeatures))), sketchtype="gauss", weight =0.1)
    dn = randn(prob.numfeatures);
    rhs = zeros(sketchsize);
    # prevx = zeros(prob.numfeatures);
    S = randn(prob.numfeatures, sketchsize);
    HS = zeros(prob.numfeatures, sketchsize);
    SHS = zeros(sketchsize, sketchsize);
    grad = zeros(prob.numfeatures); # storing the previous gradient
    avrg_dir  = randn(prob.numfeatures); #Weighted average of search direction
    if(sketchtype == "cov")
        cov_dir = Matrix{Float64}(I, prob.numfeatures, prob.numfeatures) ;
    else
        cov_dir = [];
    end
    stepsize = options.stepsize_multiplier ;
    name = string("SPIN-", sketchsize, "-", sketchtype,"-", weight);#-",options.batchsize);
    if(sketchsize == prob.numfeatures)
        name = string("Newton");
    end
    return SPIN(1, 1, name, descent_SPIN, boot_SPIN, sketchsize, dn, rhs, S, HS, SHS, grad, avrg_dir, cov_dir, weight, stepsize, sketchtype, boot_SPIN)
end