"""
    nice_sampling(probas)

Sample the indices of b (batchsize) points out of n (number of data points) at random without replacement.

# INPUTS
- **Sampling** sampling: sampling object (b-nice or independent sampling)
# OUTPUTS
- **Array{Int64}** indices: array of selected indices
"""
function nice_sampling(sampling::Sampling)
    indices = sample(1:sampling.numdata, sampling.batchsize, replace=false); # b-nice sampling
    return indices
end

"""
    independent_sampling2(probas)

Sample the indices of the randomly selected coordinates according to the independent probabilities probas. It boils down to b-independent sampling if sum(probas) = b, for instance with p_i = b/n for all data points i.

# INPUTS
- **Sampling** sampling: sampling object (b-nice or independent sampling)
# OUTPUTS
- **Array{Int64}** indices: array of selected indices
"""
function independent_sampling2(sampling::Sampling)
    D = [Bernoulli(i) for i in sampling.probas]
    indices = [rand(d) for d in D]
    indices = findall(x->x==1, vec(indices))
    return indices
end

"""
    build_sampling(type, numdata, options ; probas)

Construtor of a sampling, "independent" or "nice".

# INPUTS
- **AbstractString** type: "independent" or "nice"
- **Int64** numdata: number of data points
- **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...
- **Array{Float64}** probas: vector of probabilities of selecting each coordinate (size: number of data points)
# OUTPUTS
- **Sampling** sampling: sampling object (b-nice or independent sampling)
"""
function build_sampling(type::AbstractString, numdata::Int64, options::MyOptions ; probas::Array{Float64}=Float64[])
    if type == "nice"
        # println("nice sampling initiated")
        name = "nice"
        batchsize = options.batchsize # exact batchsize
        if batchsize != 1
            name = string(batchsize, "-", name)
        end
        probas = []
        sampleindices = nice_sampling
    elseif type == "independent"
        name = "inde"
        if isempty(probas)
            name = string("unif-", name)
            println("No probabilities given: sampling set to uniform b-independent sampling")
            batchsize = options.batchsize # average batchsize sum of the probabilities
            probas = (batchsize/numdata) * ones(numdata)
        elseif length(probas) == numdata
            batchsize = round(Int64, sum(probas)) # average batchsize sum of the probabilities
        else
            error("Incorrect probability array length")
        end
        if batchsize != 1
            name = string(batchsize, "-", name)
        end
        sampleindices = independent_sampling2
    else
        error("Undefined sampling procedure")
    end

    return Sampling(name, numdata, batchsize, probas, sampleindices)
end


## Old independent sampling
"""
    independent_sampling(probas)

Sample the indices of the randomly selected coordinates according to the independent probabilities probas. It boils down to b-independent sampling if sum(probas) = b, for instance with p_i = b/n for all data points i.

# INPUTS
- **Array{Float64}** probas: vector of probabilities of selecting each coordinate (size: number of data points)
# OUTPUTS
- **Array{Int64}** indices: array of selected indices

# Examples
```jldoctest
julia> n = 10000
julia> b = 5
julia> probas = (b/n)*ones(n) ## p_i = b/n for all i
julia> indices = independent_sampling(probas)

```
"""
function independent_sampling(probas::Array{Float64})
    D = [Bernoulli(i) for i in probas]
    indices = [rand(d) for d in D]
    indices = findall(x->x==1, vec(indices))
    return indices
end

