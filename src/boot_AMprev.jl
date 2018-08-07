function boot_AMprev(prob::Prob,method::Method, options::MyOptions)
    # if(options.embeddim==0 )
    #   embeddim= convert(Int64,min(20,ceil(prob.numfeatures/2)));
    #   embeddim = min(method.numinneriters,embeddim);
    # else
    #   embeddim = options.embeddim;
    # end
    #
    #
    # println("embeddim: ",embeddim)
    embeddim = convert(Int64, options.aux);#
    method.S = zeros(prob.numfeatures, embeddim);# contains current embedding matrix,
    method.Sold= zeros(prob.numfeatures, embeddim);# contains the old embedding matrix,
    method.diffpnt = zeros(prob.numfeatures);
    method.prevx = zeros(prob.numfeatures);
    method.gradsamp = zeros(prob.numfeatures);
    # method.prevx = zeros(prob.numfeatures,embeddim+1);# 1st position contain previous outer iterate, the 2:embedded contain the previous embedding matrix
    method.H = zeros(embeddim, embeddim);  # Store the curvature matrix STHS
    method.HS = zeros(prob.numfeatures, embeddim);
    method.HSi = zeros(prob.numfeatures, embeddim);
    method.SHS = zeros(embeddim, embeddim);  # Store the local curvature matrix STHS
    method.name = string("AMprev-", embeddim);#-",options.batchsize);
    method.gradsperiter = (embeddim + 2)*options.batchsize + (embeddim + 2)*prob.numdata/method.numinneriters + 1; #includes the cost of performing the Hessian vector product.
    method.stepmethod = descent_AMprev;
    method.aux = zeros(embeddim);
    # options.aux = embeddim; # using restol to store embedding dim!
    if(options.precondition)
        method.name = string(method.name, "-qN");
    end
    return method;
end