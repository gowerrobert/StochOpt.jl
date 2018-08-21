function load_ridge_regression(X, y::Array{Float64}, name::AbstractString, opts::MyOptions; lambda=-1, scaling="column-scaling")
    # Load ridge regression problem
    #  f(w) = 1/2n ||X'w -y||^2 + lambda/2 ||w||^2 = 1/2n sum_{i=1}^n ((X_{:i}^T w  -y_i)^2 + lambda ||w||)
    #  f_i(w) =  1/2(X_{:i}^T w  -y_i)^2 + lambda/2 ||w||^2
    #  nabla f_i =  X_{:i}(X_{:i}^T w  -y_i)^2 + lambda w

    ## standard normalization. Leave this for choosing X
    if(typeof(scaling) == String)
        datascaling = fit_apply_datascaling(X, scaling);
    elseif(typeof(scaling) == DataScaling)
        datascaling = scaling;
        apply_datascaling(X, datascaling);
    end
    name = string(name, "-", datascaling.name)
    sX = size(X);
    numfeatures = sX[1];
    numdata = sX[2];
    println("loaded ", name, " with ", numfeatures, " features and ", numdata, " data");

    if(lambda ==-1)
        if opts.regulatrizor_parameter == "1/num_data"
            lambda = 1/numdata;
        else
            lambda = mean(sum(X.^2, 1))/numdata;
            println("lambda = ", lambda);
            # display(lambda)
            #println("maximum(sum(X.^2,1)): ", maximum(sum(X.^2,1)))
        end
    end

    f_eval(x, S)                = ((1./length(S))*ridge_eval(X[:, S],y[S],x) + (lambda)*(0.5)* norm(x)^2);
    g_eval(x, S)                = ((1./length(S))*ridge_grad(X[:, S],y[S],x).+(lambda).*x);
    g_eval!(x, S, g)            = ridge_grad!(X[:, S], y[S], x, lambda, length(S), g);
    Jac_eval!(x, S, Jac)        = ridge_Jac!(X[:, S], y[S], x, lambda, S, Jac);
    scalar_grad_eval(x, S)      = ridge_scalar_grad(X[:, S], y[S], x);
    scalar_grad_hess_eval(x, S) = ridge_scalar_grad_hess(X[:, S], y[S], x);

    prob = Prob(X, y, numfeatures, numdata, 0.0, name, datascaling, f_eval, g_eval, g_eval!, Jac_eval!, scalar_grad_eval, scalar_grad_hess_eval, x->x, x->x, x->x, x->x, x->x, x->x, x->x, x->x, x->x, lambda)
    # ((1/n)X X' +lambda I)w= Xy
    # xsol = ((1/n)X X' +lambda I) \ ( (1/n)*Xy)
    xsol =(X*X' +numdata*lambda*eye(numfeatures)) \ ( X*y);
    prob.fsol = f_eval(xsol, 1:numdata);

    fsolfilename = get_fsol_filename(prob);
    save("$(fsolfilename).jld", "fsol", prob.fsol)
    return prob
end

#  f(w) = 1/2 ||X'w -y||^2
function ridge_eval(X, y::Array{Float64}, w::Array{Float64}) #,S::Array{Int64}
    return 0.5*norm(X'*w -y)^2;
end

#  sum_i nabla f_i =  sum_i X_{:i}(X_{:i}^T w  -y_i)^2
function ridge_grad(X, y::Array{Float64}, w::Array{Float64})
    return (X*(X'*w -y));
end

function ridge_scalar_grad(X, y::Array{Float64}, w::Array{Float64})
    return (X'*w -y);
end

function ridge_scalar_grad_hess(X, y::Array{Float64}, w::Array{Float64})
    return (X'*w -y), ones(length(y));
end
# (1/n)X (X'w-y) +lambda w
function  ridge_grad!(X, y::Array{Float64}, w::Array{Float64}, lambda::Float64, batch::Int64, g::Array{Float64})
    g[:]= (1/batch)*(X*(X'*w -y)) +(lambda)*w;
end

function  ridge_Jac!(X, y::Array{Float64}, w::Array{Float64}, lambda, S::Array{Int64}, Jac::Array{Float64})
    Jac[:, S] = X.*((X'*w -y)');
    # broadcast!(*,Jac[:,S],X, (y.*(t .- 1))');
    Jac[:, S] .+= (lambda).*w;
end
