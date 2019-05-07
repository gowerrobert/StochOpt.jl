## Independent sampling

"""
    independent_sampling(probas)

Sample the indices of the randomly selected coordinates according to the independent probabilities probas. It boils down to b-independent sampling if sum(probas) = b, for instance with p_i = b/n for all data points i.

# INPUTS
- **Array{Float64}** probas: vector of probabilities of selecting each coordinate (size: number of data points)
# OUTPUTS
- **Array{Int64}** indices: binary vector indicating the selected indices

# Examples
```jldoctest
julia> n = 10000
julia> b = 5
julia> probas = (b/n)*ones(n) ## p_i = b/n for all i
julia> indices = independent_sampling(probas)
julia> mean(indices)
0.005
```
"""
function independent_sampling(probas::Array{Float64})
    D = [Binomial(1, i) for i in probas]
    indices = [rand(d) for d in D]
    indices = findall(x->x==1, vec(indices))

    return indices
end

