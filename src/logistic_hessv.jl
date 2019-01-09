function  logistic_hessv(X, y::Array{Float64}, w::Array{Float64}, v::Array{Float64})
    # Hessian-vector product of logisitc function.
    # remember! y_i^2 =1, and that is why y_i^2 terms do not appear.
    Xx  = X'*w;
    yXx = y.*Xx;
    t = logistic_phi(yXx) ;
    # t = logistic_phi(y.*(X'*w));
    return X*(t.*(1-t).*(X'*v));
end

function  logistic_hessv!(X, y::Array{Float64}, w::Array{Float64}, v::Array{Float64},lambda::Float64,batch::Int64, g::Array{Float64}, Hv::Array{Float64})
    # Hessian-vector product of logisitc function.
    # remember! y_i^2 =1, and that is why y_i^2 terms do not appear.
    Xx  = X'*w;
    yXx = y.*Xx;
    t = logistic_phi(yXx) ;
    g[:] = (1/batch)*X*(y.*(t .- 1)).+(lambda).*w;
    # t = logistic_phi(y.*(X'*w));
    Hv[:] =  (1/batch)*X*(t.*(1 .- t).*(X'*v)).+ lambda.*v;
end

function  logistic_hessvv(X, y::Array{Float64}, w::Array{Float64}, v::Array{Float64})
    # Hessian-vector product of logisitc function.
    # remember! y_i^2 =1, and that is why y_i^2 terms do not appear.
    Xx  = X'*w;
    yXx = y.*Xx;
    t = logistic_phi(yXx) ;
    return ((t.*(1-t))'*((X'*v).^2));
end

function  logistic_hessD(X, y::Array{Float64}, w::Array{Float64})
    # Calculates the diagonal of the Hessian matrix.
    Xx  = X'*w;
    yXx = y.*Xx;
    t = logistic_phi(yXx);
    #return (X.^2)*(t.*(1-t));
    return sum((X.^2)'.*(t.*(1-t)),1)';
end

function  logistic_hessD!(X, y::Array{Float64}, w::Array{Float64}, lambda::Float64, batch::Int64, g::Array{Float64}, D::Array{Float64})
    # Calculates the diagonal of the Hessian matrix.
    Xx  = X'*w;
    yXx = y.*Xx;
    t = logistic_phi(yXx) ;
    g[:] = (1/batch)*X*(y.*(t .- 1)).+(lambda).*w;
    # t = logistic_phi(y.*(X'*w));
    #return (X.^2)*(t.*(1-t));
    D[:] = (1/batch)*sum((X.^2)'.*(t.*(1-t)),1)' .+ (lambda).*ones(length(w));
end
