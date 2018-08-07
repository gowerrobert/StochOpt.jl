function boot_BFGS_accel(prob::Prob, method::Method, options::MyOptions)
    method.diffpnt = zeros(prob.numfeatures);
    method.prevx = zeros(prob.numfeatures);
    method.gradsamp = zeros(prob.numfeatures); # storing the previous gradient
    # method.prevx = zeros(prob.numfeatures,embeddim+1);# 1st position contain previous outer iterate, the 2:embedded contain the previous embedding matrix
    method.HSi = zeros(prob.numfeatures, prob.numfeatures);  # stores the Y_k's
    method.SHS = eye(prob.numfeatures);  # stores the V_k's
    method.H = eye(prob.numfeatures);  # Store inverse Hessian approximation
    method.S = zeros(prob.numfeatures);  # Stores the difference between gradients
    method.HS = zeros(prob.numfeatures);  # Stores product of H*dy difference between gradients
    method.gradsperiter = 3*prob.numfeatures;

    # assuming options.embeddim = [mu, nu]
    mu = options.embeddim[1];
    nu = options.embeddim[2];
    println("(mu, nu) =  (", round(mu, 3), ", ", round(nu, 2), ")")
    method.aux = zeros(3);  # the alpha, beta, gamma parameters, in that order
    # \beta  =1 - \sqrt{\frac{\mu}{\nu}}, \gamma = \sqrt{\frac{1}{\mu \nu}}, \alpha = \frac{1}{1+\gamma\nu}.
    beta = 1 - sqrt(mu/nu);
    gamma = sqrt(1/(mu*nu));
    method.aux[1] = 1/(1 + gamma*nu);
    method.aux[2] = beta;
    method.aux[3] = gamma;
    println("alpha, beta, gamma = ", method.aux[1], ", ", method.aux[2], ", ", method.aux[3])
    method.name = string("BFGS-a-", round(mu, 2), "-", round(nu, 2));#-",options.batchsize);
    # method.gradsperiter = (embeddim+2)*options.batchsize+(embeddim+2)*prob.numdata/method.numinneriters+1; #includes the cost of performing the Hessian vector product.
    method.stepmethod = descent_BFGS_accel;
    return method;
end