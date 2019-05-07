## b-Independent sampling

using Distributions

n = 5
p = rand(n)
D = [Binomial(1, i) for i in p]

N = 1000
V = [rand(d, N) for d in D]
V = hcat(V...)'
mean(V, dims=2)
p

## At each iteration perform
n = 5
b = 1
p = (b/n)*ones(n) ## p_i = b/n for all i
D = [Binomial(1, i) for i in p]
indices = [rand(d) for d in D]

"""
    independent_sampling(probs)

Sample the indices of the randomly selected coordinates according to the independent probabilities probs.

# INPUTS
- **Array{Float64}** probs: vector of probabilities of selecting each coordinate (size: number of data points)
# OUTPUTS
- **Array{Int64}** indices: binary vector indicating the selected indices

# Examples
```jldoctest
julia> n = 10000
julia> b = 5
julia> probs = (b/n)*ones(n) ## p_i = b/n for all i
julia> indices = independent_sampling(probs)
julia> mean(indices)
0.005
```
"""
function independent_sampling(probs::Array{Float64})
    D = [Binomial(1, i) for i in probs]
    indices = [rand(d) for d in D]

    return indices
end

n = 1000
b = 5
probs = (b/n)*ones(n) ## p_i = b/n for all i
indices = independent_sampling(probs)
mean(indices)