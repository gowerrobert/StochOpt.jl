function  logistic_grad(X,y::Array{Float64},w::Array{Float64})
t = logistic_phi(y.*X'*w)
return X*(y.*(t .- 1));
end
function  logistic_grad!(X,y::Array{Float64},w::Array{Float64}, lambda::Float64, batch::Int64, g::Array{Float64})
t = logistic_phi(y.*X'*w)
g[:]= (1/batch)*X*(y.*(t .- 1)).+(lambda).*w;
end
