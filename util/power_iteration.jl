"""
    power_iteration(A ; b, numsim)

Computes the approximation largest eigenvalue of a matrix using the power iteration algorithm.
# Ref: "https://en.wikipedia.org/wiki/Power_iteration"

#INPUTS:\\
    - **Array{Float64,2}** A: squared matrix for which we want the largest eigenvalue\\
    - **Array{Float64}** b: initial vector\\
    - **Int64** numsim: number of simulations (10 seems enough for small matrices)\\
#OUTPUTS:\\
    - **Float64** maxeig: approximation of the largest eigenvalue\\
"""
function power_iteration(A::Array{Float64,2} ; b::Array{Float64}=rand(size(A,1),1), numsim::Int64=10)
    for i = 1:numsim
        b = A*b;
        b /= norm(b);
    end
    maxeig = (b'*A*b)/(b'*b);
    return maxeig[1] # 1x1 Array{Float64,2} to Float64
end

"""
    inverse_iteration(A ; b, numsim)

Computes an approximation of the smallest eigenvalue of a matrix using the power iteration algorithm.
# Ref: "https://en.wikipedia.org/wiki/Inverse_iteration"

#INPUTS:\\
    - **Array{Float64,2}** A: squared matrix for which we want to compute the smallest eigenvalue\\
    - **Array{Float64}** b: initial vector\\
    - **Int64** numsim: number of simulations (10 seems enough for small matrices)\\
#OUTPUTS:\\
    - **Float64** mineig: approximation of the smallest eigenvalue\\
"""
function inverse_iteration(A::Array{Float64,2} ; b::Array{Float64}=rand(size(A,1),1), numsim::Int64=10)
    for i = 1:numsim
        b = A \ b;
        b /= norm(b);
    end
    # mineig = (b' * (A \ b)) / (b'*b);
    # mineig = 1/mineig[1]; # 1x1 Array{Float64,2} to Float64
    mineig = (b'*b) / (b' * (A \ b));
    return mineig[1]
end

"""
    Symmetric_power_iteration(A ; b, numsim)

Computes the approximation largest eigenvalue of a symmetric matrix using the power iteration algorithm.
# Ref: "https://en.wikipedia.org/wiki/Power_iteration"

#INPUTS:\\
    - **Symmetric{Float64,Array{Float64,2}}** A: squared symmetric matrix for which we want the largest eigenvalue\\
    - **Array{Float64}** b: initial vector\\
    - **Int64** numsim: number of simulations (10 seems enough for small matrices)\\
#OUTPUTS:\\
    - **Float64** maxeig: approximation of the largest eigenvalue\\
"""
function Symmetric_power_iteration(A::Symmetric{Float64,Array{Float64,2}} ; b::Array{Float64}=rand(size(A,1),1), numsim::Int64=10)
    for i = 1:numsim
        b = A*b;
        b /= norm(b);
    end
    maxeig = (b'*A*b)/(b'*b);
    return maxeig[1] # 1x1 Array{Float64,2} to Float64
end

"""
    Symmetric_inverse_iteration(A ; b, numsim)

Computes an approximation of the smallest eigenvalue of a symmetric matrix using the power iteration algorithm.
# Ref: "https://en.wikipedia.org/wiki/Inverse_iteration"

#INPUTS:\\
    - **Symmetric{Float64,Array{Float64,2}}** A: squared symmetric matrix for which we want the largest eigenvalue\\
    - **Array{Float64}** b: initial vector\\
    - **Int64** numsim: number of simulations (10 seems enough for small matrices)\\
#OUTPUTS:\\
    - **Float64** mineig: approximation of the smallest eigenvalue\\
"""
function Symmetric_inverse_iteration(A::Symmetric{Float64,Array{Float64,2}} ; b::Array{Float64}=rand(size(A,1),1), numsim::Int64=10)
    for i = 1:numsim
        b = A \ b;
        b /= norm(b);
    end
    # mineig = (b' * (A \ b)) / (b'*b);
    # mineig = 1/mineig[1]; # 1x1 Array{Float64,2} to Float64
    mineig = (b'*b) / (b' * (A \ b));
    return mineig[1]
end