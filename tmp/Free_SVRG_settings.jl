"""
    optimal_minibatch_Free_SVRG_nice(prob::Prob, method, options::MyOptions)

Compute the optimal mini-batch size when the inner loop size m = n for the Free-SVRG algorithm with b-nice sampling.

# INPUTS
- **Int64** n: number of data samples
- **Float64** mu: strong convexity parameter of the objective function
- **Float64** L: smoothness constant of the whole objective function f
- **Float64** Lmax: max of the smoothness constant of the f_i functions
# OUTPUTS
- **Int64** minibatch_size: theoretical mini-batch size for Free-SVRG with b-nice sampling
"""
function optimal_minibatch_Free_SVRG_nice(n, mu, L, Lmax)
    b_hat = sqrt( (n*(3*Lmax-L)) / (2*(n*L-3*Lmax)) )
    b_tilda = sqrt( (n*(3*Lmax-L)) / ((n*(n-1)*mu)/(2*log(2)) - n*L + 3*Lmax) )
    flag = "none"

    if n < L/mu
        if Lmax < n*L/3
            minibatch_size = floor(Int, b_hat)
            flag = "b_hat"
        else
            minibatch_size = n
            flag = "n"
        end
    elseif L/mu <= n <= 3*Lmax/mu
        if Lmax < n*L/3
            minibatch_size = floor(Int, min(b_hat, b_tilda))
            flag = "min"
        else
            minibatch_size = floor(Int, b_tilda)
            flag = "b_tilda"
        end
    else
        minibatch_size = 1
        flag = "1"
    end

    println(flag)

    return minibatch_size
end

# """
#     minibatch_Free_SVRG_nice(prob::Prob, method, options::MyOptions)

# Compute the mini-batch size given an inner loop size m for the Free-SVRG algorithm with b-nice sampling.

# # INPUTS
# - **Int64** m: inner loop size
# - **Int64** n: number of data samples
# - **Float64** mu: strong convexity parameter of the objective function
# - **Float64** L: smoothness constant of the whole objective function f
# - **Float64** Lmax: max of the smoothness constant of the f_i functions
# # OUTPUTS
# - **Int64** minibatch_size: theoretical mini-batch size for Free-SVRG with b-nice sampling
# """
# function minibatch_Free_SVRG_nice(m, n, mu, L, Lmax)


#     return minibatch_size
# end
