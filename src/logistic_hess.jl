function logistic_hess(X, y::Array{Float64}, w::Array{Float64})
    # Hessian of pseudo-Huber function.
    # remember! y_i^2 =1, and that is why y_i^2 terms do not appear.
    Xx  = X'*w;
    yXx = y.*Xx;
    t = logistic_phi(yXx) ;
    return X*(t.*(1-t).*X');
    #Hv = X*bsxfun(@times, t.*(1-t),X'*v);
end

function logistic_hess!(X, y::Array{Float64}, w::Array{Float64}, lambda::Float64, batch::Int64, g::Array{Float64}, H::SparseMatrixCSC )
    # Hessian of pseudo-Huber function.
    # remember! y_i^2 =1, and that is why y_i^2 terms do not appear.
    Xx  = X'*w;
    yXx = y.*Xx;
    t = logistic_phi(yXx) ;
    # t = logistic_phi( y.*X'*w) ;
    g[:] = (1/batch)*X*(y.*(t .- 1)).+(lambda).*w;
    H[:] =  (1/batch)*X*(t.*(1-t).*X') + lambda* eye(length(w)) ;
    #diag(H) += (lambda);
end

function logistic_hessC(X, y::Array{Float64}, w::Array{Float64}, C::Array{Int64}, lambda::Float64, batch::Int64)
    # Hessian of pseudo-Huber function.
    # remember! y_i^2 =1, and that is why y_i^2 terms do not appear.
    Xx  = X'*w;
    yXx = y.*Xx;
    t = logistic_phi(yXx) ;
    H = (1/batch)*X*(t.*(1-t).*X[C, :]');
    H[C, :] += lambda*eye(length(C))
    return H;
    #Hv = X*bsxfun(@times, t.*(1-t),X'*v);
end

function logistic_hessC!(X, y::Array{Float64}, w::Array{Float64}, C::Array{Int64}, lambda::Float64, batch::Int64, g::Array{Float64}, HC::Array{Float64})
    # Hessian of pseudo-Huber function.
    # remember! y_i^2 =1, and that is why y_i^2 terms do not appear.
    Xx  = X'*w;
    yXx = y.*Xx;
    t = logistic_phi(yXx) ;
    # t = logistic_phi( y.*X'*w) ;
    g[:] = (1/batch)*X*(y.*(t .- 1)).+(lambda).*w;
    HC[:] = (1/batch)*X*(t.*(1-t).*X[C, :]');
    HC[C, :] += lambda*eye(length(C))
end

function logistic_hessC2(X, y::Array{Float64}, w::Array{Float64}, C::Array{Int64}, lambda::Float64, batch::Int64)
    # Hessian of pseudo-Huber function.
    # remember! y_i^2 =1, and that is why y_i^2 terms do not appear.
    Xx  = X'*w;
    yXx = y.*Xx;
    t = logistic_phi(yXx) ;
    return (1/batch)*X*(t.*(1-t).*X[C, :]') + (lambda).*eye(length(w))[:, C];
    #Hv = X*bsxfun(@times, t.*(1-t),X'*v);
end