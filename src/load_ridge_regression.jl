function load_ridge_regression(X, y::Array{Float64}, name::AbstractString, opts::MyOptions; lambda=-1, scaling="column-scaling")
    # Load ridge regression problem
    #  f(w) = 1/2n ||X'w -y||^2 + lambda/2 ||w||^2 = 1/2n sum_{i=1}^n ((X_{:i}^T w  -y_i)^2 + lambda ||w||)
    #  f_i(w) =  1/2(X_{:i}^T w  -y_i)^2 + lambda/2 ||w||^2
    #  nabla f_i =  X_{:i}(X_{:i}^T w  -y_i)^2 + lambda w

    name = string("ridge_", name);

    ## standard normalization. Leave this for choosing X
    if(typeof(scaling) == String)
        X, datascaling = fit_apply_datascaling(X, scaling);
    elseif(typeof(scaling) == DataScaling)
        datascaling = scaling;
        apply_datascaling(X, datascaling);
    end
    name = string(name, "-", datascaling.name);
    sX = size(X);
    numfeatures = sX[1];
    numdata = sX[2];

    if lambda == -1
        if(opts.regularizor_parameter == "1/num_data")
            lambda = 1/numdata;
        elseif(opts.regularizor_parameter == "normalized")
            lambda = 1/(2.0*numdata); #maximum(sum(X.^2,1))/(4.0*numdata);
            #println("maximum(sum(X.^2,1)): ", maximum(sum(X.^2,1)))
        elseif(opts.regularizor_parameter == "Lbar/n")
            lambda = mean(sum(X.^2, dims=1))/numdata; # Lbar / n
            # display(lambda)
            #println("maximum(sum(X.^2,1)): ", maximum(sum(X.^2,1)))
        else
            error("Unknown regularizor_parameter option");
        end
        name = string(name, "-regularizor-",  replace(opts.regularizor_parameter, r"[\/]" => "_"));
    elseif lambda >= 0.0
        name = string(name, "-regularizor-", replace(@sprintf("%.0e", lambda), "." => "_"));
    else
        error("lambda cannot be nonpositive (except -1)");
    end
    println("lambda = ", lambda);
    println("loaded ", name, " with ", numfeatures, " features and ", numdata, " data");

    ## To avoid very long computations when dimensions are large mu is approximated by lambda
    # mu = load_mu_str_conv(name, X, lambda) ## TO IMPLEMENT
    if numdata > 10^8 || numfeatures > 10^8
        mu = lambda
    else
        mu = get_mu_str_conv(X, lambda) # mu = minimum(sum(prob.X.^2, 1)) + prob.lambda;
    end
    L = get_LC(X, lambda, collect(1:numdata)); # L = eigmax(prob.X*prob.X')/n + prob.lambda;
    Li_s = get_Li(X, lambda);
    Lmax = maximum(Li_s); # Lmax = maximum(sum(prob.X.^2, 1)) + prob.lambda;
    Lbar = mean(Li_s); # = mean(sum(X.^2, dims=1))

    f_eval(x, S)                = ((1 ./ length(S))*ridge_eval(X[:, S], y[S], x) + lambda*0.5*norm(x)^2);
    g_eval(x, S)                = ((1 ./ length(S))*ridge_grad(X[:, S], y[S], x) + lambda*x);
    g_eval!(x, S, g)            = ridge_grad!(X[:, S], y[S], x, lambda, length(S), g);
    Jac_eval!(x, S, Jac)        = ridge_Jac!(X[:, S], y[S], x, lambda, S, Jac);
    scalar_grad_eval(x, S)      = ridge_scalar_grad(X[:, S], y[S], x);
    scalar_grad_hess_eval(x, S) = ridge_scalar_grad_hess(X[:, S], y[S], x);

    prob = initiate_Prob(X=X, y=y, numfeatures=numfeatures, numdata=numdata, name=name, datascaling=datascaling, f_eval=f_eval, g_eval=g_eval, g_eval_mutating=g_eval!, Jac_eval_mutating=Jac_eval!, scalar_grad_eval=scalar_grad_eval, scalar_grad_hess_eval=scalar_grad_hess_eval, lambda=lambda, mu=mu, L=L, Lmax=Lmax, Lbar=Lbar)

    ## Try to load the solution of the problem, if already computed
    load_fsol!(opts, prob);

    if prob.fsol == 0.0
        println("Computing and saving the exact solution of the problem")
        get_fsol_ridge!(prob); ## getting and saving approximation of the solution fsol
    end

    # fsolfilename = get_fsol_filename(prob);
    # save("$(fsolfilename).jld", "fsol", prob.fsol)
    return prob
end

#  f(w) = 1/2 ||X'w -y||^2
function ridge_eval(X, y::Array{Float64}, w::Array{Float64}) #,S::Array{Int64}
    return 0.5*norm(X'*w - y)^2;
end

#  sum_i nabla f_i =  sum_i X_{:i}(X_{:i}^T w  -y_i)^2
function ridge_grad(X, y::Array{Float64}, w::Array{Float64})
    return (X*(X'*w - y));
end

function ridge_scalar_grad(X, y::Array{Float64}, w::Array{Float64})
    return (X'*w - y);
end

function ridge_scalar_grad_hess(X, y::Array{Float64}, w::Array{Float64})
    return (X'*w - y), ones(length(y));
end
# (1/n)X (X'w-y) +lambda w
function ridge_grad!(X, y::Array{Float64}, w::Array{Float64}, lambda::Float64, batch::Int64, g::Array{Float64})
    g[:] = (1/batch)*(X*(X'*w - y)) + lambda*w;
end

function ridge_Jac!(X, y::Array{Float64}, w::Array{Float64}, lambda::Float64, S::Array{Int64}, Jac::Array{Float64})
    Jac[:, S] = X.*((X'*w - y)');
    # broadcast!(*,Jac[:,S],X, (y.*(t .- 1))');
    Jac[:, S] .+= lambda*w;

    ## Why not ?
    # Jac[:, S] = X.*((X'*w - y)') .+ lambda*w;
end

"""
    get_fsol_ridge!(prob)

Compute and save the exact solution of the given ridge regression problem. The solution is obtained by computing:\\
xsol = ( (1/n)X X' + lambda I )^(-1) Xy

#INPUTS:\\
    - **Prob** prob: ridge regression problem\\
#OUTPUTS:\\
"""
function get_fsol_ridge!(prob)
    # ((1/n)X X' + lambda I) w = Xy
    # xsol = ( (1/n)X X' + lambda I ) \ Xy

    ## Computation of fsol
    xsol = (prob.X*prob.X' + prob.numdata*prob.lambda*Matrix(1.0I, prob.numfeatures, prob.numfeatures)) \ (prob.X*prob.y); # no more 'eye(numfeatures)' in julia 0.7
    prob.fsol = prob.f_eval(xsol, 1:prob.numdata);
    ## Saving fsol
    fsolfilename = get_fsol_filename(prob);
    save("$(fsolfilename).jld", "fsol", prob.fsol)
end