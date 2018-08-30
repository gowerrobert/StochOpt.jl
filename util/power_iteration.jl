# function power_iteration(A::Array{Float64} ; b::Array{Float64}=rand(size(A,1),1), numsim::Int64=10)
function power_iteration(A::Symmetric{Float64,Array{Float64,2}} ; b::Array{Float64}=rand(size(A,1),1), numsim::Int64=10)
    # power_iteration computes the largest eigenvalue of a matrix using the power iteration algorithm
    ## INPUT
    #   A: squar matrix for which we want the largest eigenvalue
    #   b: initial vector
    #   numsim: number of simulations (10 is often enough for small matrices?)
    ## OUTPUT
    #   maxeig: largest eigenvalue of A
    # Ref: "https://en.wikipedia.org/wiki/Power_iteration"
    for i = 1:numsim
        b = (A*b);
        b = b/norm(b);
    end
    maxeig = (b'*A*b)/(b'*b);
    return maxeig[1] # 1x1 Array{Float64,2} to Float64
end