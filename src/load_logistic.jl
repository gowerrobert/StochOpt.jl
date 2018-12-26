function load_logistic(probname::AbstractString, datapath::AbstractString, opts::MyOptions; scaling="column-scaling")
    # Load logistic regression problem
    #try
    name = string("lgstc_", probname);  # maybe add different regularizor later opts.regularizor
    datafile = string(datapath, probname); println("loading:  ", datafile)
    X, y = loadDataset(datafile);
    stdX = std(X, 2);
    #replace 0 in std by 1 incase there is a constant feature
    ind = (0 .== stdX); stdX[ind] = 1.0; # Testing for a zero std # julia 0.7
    X[:, :] = (X.-mean(X,2))./stdX; # Centering and scaling the data.
    X = [X; ones(size(X, 2))'];
    # Underdevelopment: a Datascaling structure
    datascaling = DataScaling([], [], [], "..");
    # if(typeof(scaling) == String)
    #   datascaling = fit_apply_datascaling(X, scaling);
    # elseif(typeof(scaling) == DataScaling)
    #   datascaling = scaling;
    #   apply_datascaling(X, datascaling);
    # end
    # if(datascaling.name != "column-scaling")
    #   name = string(name,"-",datascaling.name)
    # end
    sX = size(X);
    numfeatures = sX[1];
    numdata = sX[2];
    #Transforming y to the binary to -1 and 1 representation
    y[find(x->x==minimum(y), y)] = -1;
    y[find(x->x==maximum(y), y)] = 1;

    println("loaded ", name, " with ", numfeatures, " features and ", numdata, " data");
    if opts.regularizor_parameter == "1/num_data"
        lambda = 1/numdata;
    elseif opts.regularizor_parameter == "normalized"
        lambda = 1/(2.0*numdata) #maximum(sum(X.^2,1))/(4.0*numdata);
        #println("maximum(sum(X.^2,1)): ", maximum(sum(X.^2,1)))
    else
        lambda = opts.regularizor_parameter;
    end
    println("lambda = ", lambda);

    mu = get_mu_str_conv(X, lambda); # mu = minimum(sum(prob.X.^2, 1)) + prob.lambda;
    L = get_LC(X, lambda, collect(1:numdata)); # L = eigmax(prob.X*prob.X')/n + prob.lambda;
    Li_s = get_Li(X, lambda);
    Lmax = maximum(Li_s); # Lmax = maximum(sum(prob.X.^2, 1)) + prob.lambda;
    Lbar = mean(Li_s);

    # if opts.regularizor =="huber"
    #         f_eval(x,S) =  (1./length(S))*logistic_eval(Xt,y,x,S)+(reg)* huber_eval(x,opts.hubermu);
    #         g_eval(x,S) = ((1./length(S))*logistic_grad_sub(Xt,y,x,S)+(reg)*huber_grad(x,opts.hubermu));
    #         Hess_opt(x,S,v) = ((1./length(S))*logistic_hessv_sub(Xt,y,x,S,v)+(reg)*bsxfun(@times, huber_hess_kimon(x,opts.hubermu), v) );
    #if opts.regularizor =="L2"# is the default
    f_eval(x, S) = ((1. / length(S))*logistic_eval(X[:,S],y[S],x)+(lambda)*(0.5)* norm(x)^2); # julia 0.7
    g_eval(x, S) = ((1. / length(S))*logistic_grad(X[:,S],y[S],x).+(lambda).*x); # julia 0.7
    g_eval!(x, S, g) =              logistic_grad!(X[:,S],y[S],x, lambda,length(S),g);
    Jac_eval!(x, S, Jac) =           logistic_Jac!(X[:,S],y[S],x, lambda,S, Jac);
    scalar_grad_eval(x, S) = logistic_scalar_grad(X[:,S],y[S],x)
    scalar_grad_hess_eval(x, S) = logistic_scalar_grad_hess(X[:,S],y[S],x)
    Hess_eval(x, S) = ((1 ./ length(S))*logistic_hess(X[:,S],y[S],x).+ (lambda).*eye(numfeatures)); # julia 0.7
    Hess_eval!(x, S, g, H) =              logistic_hess!(X[:,S],y[S],x,lambda,length(S),g,H) ;
    Hess_C(x, S, C) =                  logistic_hessC(X[:,S],y[S],x,C,lambda,length(S)); # .+ (lambda).*eye(numfeatures)[:,C]not great solution on the identity
    Hess_C!(x, S, C, g, HC) = logistic_hessC!(X[:,S],y[S],x,C,lambda,length(S),g,HC);
    Hess_C2(x, S, C) = logistic_hessC(X[:,S],y[S],x,C,lambda,length(S));
    Hess_opt(x, S, v)     = ( (1 ./ length(S))*logistic_hessv(X[:,S],y[S],x,v).+ (lambda).*v); # julia 0.7
    Hess_opt!(x, S, v, g, Hv) =                  logistic_hessv!(X[:,S],y[S],x,v,lambda,length(S),g,Hv);
    Hess_D(x, S) = ((1 ./ length(S))*logistic_hessD(X[:,S],y[S],x).+ (lambda).*ones(numfeatures)); # julia 0.7
    Hess_D!(x, S, g, D) =              logistic_hessD!(X[:,S],y[S],x,lambda,length(S),g,D);
    #Hess_vv(x, S, v) = ((1./length(S))*logistic_hessvv(X[:,S],y[S],x,v).+ (lambda).*v'*v);
    #else
    #        println("Choose regularizor huber or L2");
    #        error("Unknown regularizor"+ opts.regularizor);
    #end 

    prob = Prob(X, y, numfeatures, numdata, 0.0, name, datascaling, f_eval, g_eval, g_eval!, Jac_eval!, scalar_grad_eval, scalar_grad_hess_eval,
        Hess_eval, Hess_eval!, Hess_opt, Hess_opt!, Hess_D, Hess_D!, Hess_C, Hess_C!, Hess_C2, lambda, mu, L, Lmax, Lbar)
    return prob
end

"""
    load_logistic_from_matrices(X, y, name, opts; scaling="column-scaling")

Load a logistic regression problem. The option input sets the regularization parameter lambda.

#INPUTS:\\
    - nxd X: design matrix\\
    - nx1 **Array{Float64}** y: label vector\\
    - **AbstractString** name: name of the data set\\
    - **MyOptions** opts: selected options (see "StochOpt.jl")\\
    - scaling: scaling procedure () \\
#OUTPUTS:\\
    - **Prob** prob: considered problem, here logistic regression
"""
function load_logistic_from_matrices(X, y::Array{Float64}, name::AbstractString, opts::MyOptions; lambda=-1, scaling="column-scaling")
    # Load logistic regression problem

    name = string("lgstc_", name);
    
    ## standard normalization. Leave this for choosing X
    datascaling = DataScaling([], [], [], "..");
    if(typeof(scaling) == String)
        datascaling = fit_apply_datascaling(X, scaling);
    elseif(typeof(scaling) == DataScaling)
        datascaling = scaling;
        apply_datascaling(X, datascaling);
    end
    name = string(name, "-", datascaling.name)

    # stdX = std(X, dims=2);
    # # Replace 0 in std by 1 incase there is a constant feature
    # ind = (0 .== stdX); # Testing for a zero std # julia 0.7
    # stdX[ind] .= 1.0;
    # X[:, :] = (X.-mean(X, dims=2))./stdX; # Centering and scaling the data.
    # X = [X; ones(size(X, 2))'];

    # # Under development: a Datascaling structure
    # datascaling = DataScaling([], [], [], "..");
    # # if(typeof(scaling) == String)
    # #   datascaling = fit_apply_datascaling(X, scaling);
    # # elseif(typeof(scaling) == DataScaling)
    # #   datascaling = scaling;
    # #   apply_datascaling(X, datascaling);
    # # end
    # # if(datascaling.name != "column-scaling")
    # #   name = string(name,"-",datascaling.name)
    # # end

    sX = size(X);
    numfeatures = sX[1];
    numdata = sX[2];
    #Transforming y to the binary to -1 and 1 representation
    y[findall(x->x==minimum(y), y)] .= -1;
    y[findall(x->x==maximum(y), y)] .= 1;

    println("loaded ", name, " with ", numfeatures, " features and ", numdata, " data");
    if(lambda == -1)
        if(opts.regularizor_parameter == "1/num_data")
            lambda = 1/numdata;
        elseif(opts.regularizor_parameter == "normalized")
            lambda = 1/(2.0*numdata) #maximum(sum(X.^2,1))/(4.0*numdata);
            #println("maximum(sum(X.^2,1)): ", maximum(sum(X.^2,1)))
        elseif(opts.regularizor_parameter == "Lbar/n")
            lambda = mean(sum(X.^2, dims=1))/numdata; # Lbar / n # julia 0.7
            # display(lambda)
            #println("maximum(sum(X.^2,1)): ", maximum(sum(X.^2,1)))
        else
            error("Unknown regularizor_parameter option");
        end
    end
    println("lambda = ", lambda, "\n");

    ## To avoid very long computations when dimensions are large mu is approximated by lambda
    if numdata > 10000 || numfeatures > 10000
        mu = lambda;
    else
        mu = get_mu_str_conv(X, lambda); # mu = minimum(sum(prob.X.^2, 1)) + prob.lambda;
    end
    L = get_LC(X, lambda, collect(1:numdata)); # L = eigmax(prob.X*prob.X')/n + prob.lambda;
    Li_s = get_Li(X, lambda);
    Lmax = maximum(Li_s); # Lmax = maximum(sum(prob.X.^2, 1)) + prob.lambda;
    Lbar = mean(Li_s);

    # if opts.regularizor =="huber"
    #         f_eval(x,S) =  (1./length(S))*logistic_eval(Xt,y,x,S)+(reg)* huber_eval(x,opts.hubermu);
    #         g_eval(x,S) = ((1./length(S))*logistic_grad_sub(Xt,y,x,S)+(reg)*huber_grad(x,opts.hubermu));
    #         Hess_opt(x,S,v) = ((1./length(S))*logistic_hessv_sub(Xt,y,x,S,v)+(reg)*bsxfun(@times, huber_hess_kimon(x,opts.hubermu), v) );
    #if opts.regularizor =="L2"# is the default
    f_eval(x, S)                = ((1. / length(S))*logistic_eval(X[:,S], y[S], x) + (lambda)*(0.5)*norm(x)^2); # julia 0.7
    g_eval(x, S)                = ((1. / length(S))*logistic_grad(X[:,S], y[S], x) .+ (lambda).*x); # julia 0.7
    g_eval!(x, S, g)            = logistic_grad!(X[:,S], y[S], x, lambda, length(S), g);
    Jac_eval!(x, S, Jac)        = logistic_Jac!(X[:,S], y[S], x, lambda, S, Jac);
    scalar_grad_eval(x, S)      = logistic_scalar_grad(X[:,S], y[S], x)
    scalar_grad_hess_eval(x, S) = logistic_scalar_grad_hess(X[:,S], y[S], x)
    Hess_eval(x, S)             = ((1 ./ length(S))*logistic_hess(X[:,S], y[S], x).+ (lambda).*eye(numfeatures)); # julia 0.7
    Hess_eval!(x, S, g, H)      = logistic_hess!(X[:,S], y[S], x, lambda, length(S), g, H) ;
    Hess_C(x, S, C)             = logistic_hessC(X[:,S], y[S], x, C, lambda, length(S)); # .+ (lambda).*eye(numfeatures)[:,C]not great solution on the identity
    Hess_C!(x, S, C, g, HC)     = logistic_hessC!(X[:,S], y[S], x, C, lambda, length(S), g, HC);
    Hess_C2(x, S, C)            = logistic_hessC(X[:,S], y[S], x, C, lambda, length(S));
    Hess_opt(x, S, v)           = ((1 ./ length(S))*logistic_hessv(X[:,S], y[S], x, v) .+ (lambda).*v); # julia 0.7
    Hess_opt!(x, S, v, g, Hv)   = logistic_hessv!(X[:,S], y[S], x, v, lambda, length(S), g, Hv);
    Hess_D(x, S)                = ((1 ./ length(S))*logistic_hessD(X[:,S], y[S], x) .+ (lambda).*ones(numfeatures)); # julia 0.7
    Hess_D!(x, S, g, D)         = logistic_hessD!(X[:,S], y[S], x, lambda, length(S), g, D);
    # Hess_vv(x, S, v)            = ((1./length(S))*logistic_hessvv(X[:,S], y[S], x, v) .+ (lambda).*v'*v);
    #else
    #        println("Choose regularizor huber or L2");
    #        error("Unknown regularizor"+ opts.regularizor);
    #end

    prob = Prob(X, y, numfeatures, numdata, 0.0, name, datascaling, f_eval, g_eval, g_eval!, Jac_eval!, scalar_grad_eval, scalar_grad_hess_eval,
        Hess_eval, Hess_eval!, Hess_opt, Hess_opt!, Hess_D, Hess_D!, Hess_C, Hess_C!, Hess_C2, lambda, mu, L, Lmax, Lbar)

    ## Try to load the solution of the problem, if already computed
    load_fsol!(opts, prob);
    
    if prob.fsol == 0.0
        println("Computing and saving the solution of the problem")
        get_fsol_logistic!(prob); ## getting and saving approximation of the solution fsol
    end
    
    return prob
end

"""
    get_fsol_logistic!(prob)

Compute and save an approximation of the solution of the given logistic regression problem.
The solution is obtained by running a BFGS and an accelerated BFGS algorithm.

#INPUTS:\\
    - **Prob** prob: logistic regression problem\\
#OUTPUTS:\\
"""
function get_fsol_logistic!(prob)
    if prob.numdata < 10000 || prob.numfeatures < 10000 # WARNING: symbols inverted
        options = set_options(tol=10.0^(-16.0), skip_error_calculation=10^4, exacterror=false, max_iter=10^8, 
                              max_time=60.0*60.0*3.0, max_epocs=10^5, repeat_stepsize_calculation=true, rep_number=2);
        println("Dimensions are too large too compute the solution using BFGS, using SVRG instead")
        ## Running SVRG
        # options.batchsize = 1;
        options.batchsize = prob.numdata;
        method_name = "SVRG";
        output = minimizeFunc_grid_stepsize(prob, method_name, options);

        ## Setting the true solution as the smallest of both
        prob.fsol = minimum(output.fs);
    else
        options = set_options(tol=10.0^(-16.0), skip_error_calculation=20, exacterror=false, max_iter=10^8, 
                              max_time=60.0*60.0*3.0, max_epocs=500, repeat_stepsize_calculation=true, rep_number=2);
        ## Running BFGS
        options.batchsize = prob.numdata;
        method_name = "BFGS";
        output = minimizeFunc_grid_stepsize(prob, method_name, options);

        ## Running accelerated BFGS
        options.embeddim = [prob.numdata, 0.01];  #[0.9, 5];
        method_name = "BFGS_accel";
        output1 = minimizeFunc_grid_stepsize(prob, method_name, options);
        OUTPUTS = [output; output1];

        ## Ploting the two solutions
        gr()# gr() pyplot() # pgfplots() #plotly()
        plot_outputs_Plots(OUTPUTS, prob, options);

        ## Setting the true solution as the smallest of both
        prob.fsol = minimum([output.fs output1.fs]);#min(output.fs[end],fsol);
    end

    ## Saving the solution in a JLD file
    # fsolfilename = get_fsol_filename(prob);
    # save("$(fsolfilename).jld", "fsol", prob.fsol)
end