function logistic_grad(X, y::Array{Float64}, w::Array{Float64})
    t = logistic_phi(y.*X'*w);
    return X*(y.*(t .- 1));
end

function logistic_grad!(X, y::Array{Float64}, w::Array{Float64}, lambda::Float64, batch::Int64, g::Array{Float64})
    t = logistic_phi(y.*X'*w);
    g[:] = (1/batch)*X*(y.*(t .- 1)).+(lambda).*w;
end

function logistic_scalar_grad(X, y::Array{Float64}, w::Array{Float64})
    Xx  = X'*w;
    yXx = y.*Xx;
    t = logistic_phi(yXx) ;
    return  y.*(t .- 1);
end

function logistic_scalar_grad_hess(X, y::Array{Float64}, w::Array{Float64})
    Xx  = X'*w;
    yXx = y.*Xx;
    t = logistic_phi(yXx) ;
    return  y.*(t .- 1), t.*(1 .- t);
end

function logistic_Jac!(X, y::Array{Float64}, w::Array{Float64}, lambda::Float64, S::Array{Int64}, Jac::Array{Float64})
    t = logistic_phi(y.*X'*w);
    Jac[:, S] = X.*(y.*(t .- 1))';
    # broadcast!(*,Jac[:,S],X, (y.*(t .- 1))');
    Jac[:, S] .+= (lambda).*w;
end
#
#  fi=  phi(x_i, w) + lambda ||w||, nabla fi = x_i phi'(x_i, w) + lambda w,
#  Jac = [x_1 ... x_n]diag(phi'(x_i, w)) .+ lambda w
