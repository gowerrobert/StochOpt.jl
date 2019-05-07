## Independent sampling

"""
    independent_sampling(probs)

Sample the indices of the randomly selected coordinates according to the independent probabilities probs. It boils down to b-independent sampling if sum(probs) = b, for instance with p_i = b/n for all data points i.

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