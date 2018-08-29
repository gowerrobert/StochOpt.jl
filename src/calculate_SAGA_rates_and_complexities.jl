function calculate_rate_SAGA_nice(prob::Prob, method, options::MyOptions)

    # Calculate the

end

function get_Li(prob::Prob)
    # Return an array of the Li_s
    n = prob.numdata;
    Li_s = zeros(1, n);
    for i = 1:n
        Li_s[i] = prob.X[:,i]'*prob.X[:,i] + prob.lambda;
    end
    return Li_s
end

function get_LC(prob::Prob, C)
    # if( length(C) < prob.features)
        return eigmax(Symmetric(full(prob.X[:, C]'*prob.X[:, C])))/length(C) + prob.lambda #
    # else
    #     return eigmax(Symmetric(prob.X[:,C]*prob.X[:,C]'))/length(C) +prob.lambda;
    # end
end

function get_expected_smoothness_cst(prob::Prob, tau::Int64)
    ## Computing the expected smoothness constant for a given minibatch size tau
    n = prob.numdata;
    Csets = combinations(1:n, tau);
    Ls = zeros(1, n);
    c1 = binomial(n-1, tau-1);
    # Iteration is on the sets then saving is done for corresponding indices
    # It's another way of counting than in the definition of the expected smoothness constant
    # (first an iteration over the indices, then an iteration over the sets containing the picked index)
    for C in Csets
        for i in C
            Ls[i] = Ls[i] + (1/c1)*get_LC(prob, C);
        end
    end
    expsmoothcst = maximum(Ls);
    return expsmoothcst
end

function calculate_complex_SAGA_partition_optimal(prob::Prob, method, options::MyOptions)

    # Calculate the
    # \left(n +4\frac{ \bar{L}}{\mu}  \right)  \log\left(\frac{1}{\epsilon} \right).
    prob.numdata + 4*method.L/method.mu;

    # \epsilon  = e^(-(mu k)/(4( mu n/4 +\bar{L} )))
end

function calculate_complex_SAGA_nice(prob::Prob, options::MyOptions, tauseq::Vector{Int64}=1:prob.numdata)
    # "Writing Vector{Float64} is equivalent to writing Array{Float64,1}"

    # Calculating the expected smoothness constants for nice mini-batch SAGA
    # for all possible mini-batch size from 1 (SGD) to n (gradient descent)
    # tauseq : array of the mini-batch sizes for which the complexity is computed

    n = prob.numdata;
    numtau = length(tauseq);
    itercomp = zeros(1, numtau);
    Lsides = zeros(1, numtau);
    Rsides = zeros(1, numtau);

    # Computing smallest and largest eigenvalues of the design matrix
    mu = get_mu_str_conv(prob);
    Lmax = maximum(sum(prob.X.^2, 1)) + prob.lambda;

    # For each mini-batch size computing the expected smoothness constant and then the iteration complexity
    for tauidx = 1:numtau
        tau = tauseq[tauidx];
    # for tau in tauseq
        # display(string("Calculating for tau =", tau))
        print("Calculating for tau = ", tau, "\n");

        ## Computing the right-hand side term of the complexity from RMG, Richtarik and Bach(2018), eq. (103)
        # Rsides[tau] = (((n-tau)/(tau*(n-1)))*Lmax + (mu/4)*(n/tau))*(4/mu);
        Rsides[tauidx] = (((n-tau)/(tau*(n-1)))*Lmax + (mu/4)*(n/tau))*(4/mu);

        ## Computing the right-hand side term, i.e. the expected smoothness constant
        Lsides[tauidx] = get_expected_smoothness_cst(prob, tau)*(4/mu);
        itercomp[tauidx] = max(Lsides[tauidx], Rsides[tauidx]);
    end
    itercomp = itercomp;

    return itercomp, Lsides, Rsides
end

function calculate_complex_Hofmann(prob::Prob, options::MyOptions)
    n = prob.numdata;
    mu = get_mu_str_conv(prob);
    Lmax = maximum(sum(prob.X.^2,1)) + prob.lambda;
    itercomp = zeros(1, n);
    for tau = 1:n
        K = 4*tau*Lmax/(n*mu);
        itercomp[tau] = (n/tau)*(1 + K +sqrt(1+K^2))/2;
    end
    return itercomp
end

function get_mu_str_conv(prob::Prob)
    if(prob.numfeatures < prob.numdata)
        mu = eigmin(full(prob.X*prob.X'))/prob.numdata + prob.lambda;
    else
        mu = eigmin(full(prob.X'*prob.X))/prob.numdata + prob.lambda;
    end
    return mu
end

# Sampling over a partition
# \frac{n}{\tau}+4\frac{ \max_{C \in \mathcal{G}} L_{C}}{\mu}  \right)  .
#
# Sampling with replacement
# \frac{4}{\mu}\max \left\{ \overline{L}_{\cal G}, \,
# \frac{n-\tau+1}{n\tau}L_{\max}    +\frac{\mu}{4} \frac{n}{\tau}\right\}.