
function calculate_rate_SAGA_nice(prob::Prob,method,options::MyOptions)

    # Calculate the


end

function get_LC(prob::Prob, C)
    # if( length(C) < prob.features)
        return eigmax(Symmetric(full(prob.X[:,C]'*prob.X[:,C])))/length(C) +prob.lambda; #
    # else
    #     return eigmax(Symmetric(prob.X[:,C]*prob.X[:,C]'))/length(C) +prob.lambda;
    # end
end

function calculate_complex_SAGA_partition_optimal(prob::Prob,method,options::MyOptions)

    # Calculate the
    # \left(n +4\frac{ \bar{L}}{\mu}  \right)  \log\left(\frac{1}{\epsilon} \right).
    prob.numdata+ 4*method.L/method.mu;

    # \epsilon  = e^(-(mu k)/(4( mu n/4 +\bar{L} )))
end


function calculate_complex_SAGA_nice(prob::Prob,options::MyOptions)
    # calculating the expected smoothness constants for nice mini-batch SAGA
    n = prob.numdata;
    mu = get_mu_str_conv(prob);
    Lmax = maximum(sum(prob.X.^2,1))+prob.lambda;
    itercomp = zeros(1,n);
    Lsides = zeros(1,n);
    Rsides = zeros(1,n);
    for tau =1:n
        display(string("Calculating for tau =", tau))
        Csets = combinations(1:n,tau);
        Rsides[tau] = (((n-tau)/(tau*(n-1)))*Lmax + (mu/4)*(n/tau))*(4/mu);
        Ls = zeros(1,n);
        c1 = binomial(n-1,tau-1);
        for C in Csets
            for i in C
                Ls[i] =  Ls[i] +(1/c1)*get_LC(prob, C);
            end
        end
        Lsides[tau] = maximum(Ls)*(4/mu);
        itercomp[tau] = max(Lsides[tau],Rsides[tau]);
    end
    itercomp = itercomp;

    return itercomp, Lsides, Rsides
end

function calculate_complex_Hofmann(prob::Prob,options::MyOptions)
    n = prob.numdata;
    mu = get_mu_str_conv(prob);
    Lmax = maximum(sum(prob.X.^2,1))+prob.lambda;
    itercomp = zeros(1,n);
    for tau =1:n
        K =  4*tau*Lmax/(n*mu);
        itercomp[tau] = (n/tau)*(1 + K +sqrt(1+K^2))/2;
    end
    return itercomp
end


function get_mu_str_conv(prob::Prob)
    if(prob.numfeatures< prob.numdata)
        mu = eigmin(full(prob.X*prob.X'))/prob.numdata +prob.lambda;
    else
        mu = eigmin(full(prob.X'*prob.X))/prob.numdata +prob.lambda;
    end
return mu
end

# Sampling over a partition
# \frac{n}{\tau}+4\frac{ \max_{C \in \mathcal{G}} L_{C}}{\mu}  \right)  .
#
# Sampling with replacement
# \frac{4}{\mu}\max \left\{ \overline{L}_{\cal G}, \,
# \frac{n-\tau+1}{n\tau}L_{\max}    +\frac{\mu}{4} \frac{n}{\tau}\right\}.
