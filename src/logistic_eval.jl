function logistic_eval(X::Array{Float64},y::Array{Float64},w::Array{Float64}) #,S::Array{Int64}
return -sum(log.(logistic_phi((y).*(X'*w))));
end


function logistic_eval(X::SparseMatrixCSC{Float64,Int64} ,y::Array{Float64},w::Array{Float64}) #,S::Array{Int64}
return -sum(log.(logistic_phi((y).*(X'*w))));
end
