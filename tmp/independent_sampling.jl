## b-Independent sampling

using Distributions

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

## For b-independent sampling, at each descent step, do the following
n = 1000
b = 5
probs = (b/n)*ones(n) ## p_i = b/n for all i
indices = independent_sampling(probs)
mean(indices)