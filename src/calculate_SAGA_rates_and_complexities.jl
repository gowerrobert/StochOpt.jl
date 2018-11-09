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
    # println("full")
    # full(prob.X[:, C]'*prob.X[:, C])
    # println("Symmetric")
    # Symmetric(full(prob.X[:, C]'*prob.X[:, C]))
    # println("Eigmax")
    # eigmax(Symmetric(full(prob.X[:, C]'*prob.X[:, C])))
    LC = 0;
    if(length(C) < prob.numfeatures)
        try
            LC = eigmax(Symmetric(full(prob.X[:, C]'*prob.X[:, C])))/length(C) + prob.lambda;
        catch loaderror # Uses power iteration if eigmax fails
            # println("Using power iteration instead of eigmax which returns the following error: ", loaderror);
            LC = power_iteration(Symmetric(full(prob.X[:, C]'*prob.X[:, C])))/length(C) + prob.lambda;
        end
    else
        try
            LC = eigmax(Symmetric(full(prob.X[:,C]*prob.X[:,C]')))/length(C) + prob.lambda;
        catch loaderror # Uses power iteration if eigmax fails
            # println("Using power iteration instead of eigmax which returns the following error: ", loaderror);
            LC = power_iteration(Symmetric(full(prob.X[:, C]*prob.X[:, C]')))/length(C) + prob.lambda;
        end
    end
    return LC
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
        Ls[C] = Ls[C] + (1/c1)*get_LC(prob, C); # Implementation without inner loop
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


"""
    get_expected_smoothness_bounds(prob::Prob, datathreshold::Int64=24)

Compute two upper-bounds of the expected smoothness constant (simple and Bernstein), 
a heuristic estimation of it and its exact value (if there are few data points) for each mini-batch size ``τ`` from 1 to n.

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)\\
    - **Int64** datathreshold: number of data below which exact smoothness constant is computed\\
#OUTPUTS:\\
    - nx1 **Array{Float64,2}** simplebound: simple upper bound\\
    - nx1 **Array{Float64,2}** bernsteinbound: Bernstein upper bound\\
    - nx1 **Array{Float64,2}** heuristicbound: heuristic estimation\\
    - nx1 **Array{Float64,2}** or **Void** expsmoothcst: exact expected smoothness constant\\
"""
function get_expected_smoothness_bounds(prob::Prob, datathreshold::Int64=24);
    n = prob.numdata;
    d = prob.numfeatures;
    
    if(n <= datathreshold)
        computeexpsmooth = true;
        expsmoothcst = zeros(n, 1);
    else # if n is too large we do not compute the exact expected smoothness constant
        computeexpsmooth = false;
        expsmoothcst = nothing; # equivalent of "None"
        println("The number of data is to large to compute the exact expected smoothness constant");
    end

    ### COMPUTING DIVERSE SMOOTHNESS CONSTANTS ###
    L = get_LC(prob, collect(1:n)); # WARNING: it should not be recomputed every time
    Li_s = get_Li(prob);
    Lmax = maximum(Li_s);
    Lbar = mean(Li_s);

    ### COMPUTING THE UPPER-BOUNDS OF THE EXPECTED SMOOTHNESS CONSTANT ###
    simplebound = zeros(n, 1);
    heuristicbound = zeros(n, 1);
    bernsteinbound = zeros(n, 1);
    for tau = 1:n
        print("Calculating bounds for tau = ", tau, "\n");
        if(n <= datathreshold)
            expsmoothcst[tau] = get_expected_smoothness_cst(prob, tau);
        end
        leftcoeff = (n*(tau-1))/(tau*(n-1));
        rightcoeff = (n-tau)/(tau*(n-1));
        simplebound[tau] = leftcoeff*Lbar + rightcoeff*Lmax;
        heuristicbound[tau] = leftcoeff*L + rightcoeff*Lmax;
        bernsteinbound[tau] = 2*leftcoeff*L + (rightcoeff + (4*log(d))/(3*tau))*Lmax;
    end

    return simplebound, bernsteinbound, heuristicbound, expsmoothcst
end

"""
    get_stepsize_bounds(prob::Prob, datathreshold::Int64=24)

Compute upper bounds of the stepsize based on the simple bound, the Bernstein bound, our heuristic 
and the exact expected smoothness constant for each mini-batch size ``τ`` from 1 to n.

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)\\
    - nx1 **Array{Float64,2}** simplebound: simple upper bound\\
    - nx1 **Array{Float64,2}** bernsteinbound: Bernstein upper bound\\
    - nx1 **Array{Float64,2}** heuristicbound: heuristic estimation\\
    - nx1 **Array{Float64,2}** or **Void** expsmoothcst: exact expected smoothness constant\\
#OUTPUTS:\\
    - nx1 **Array{Float64,2}** simplestepsize: lower bound of the stepsize corresponding to the simple upper bound\\
    - nx1 **Array{Float64,2}** bernsteinstepsize: lower bound of the stepsize corresponding to the Bernstein upper bound\\
    - nx1 **Array{Float64,2}** heuristicstepsize: lower bound of the stepsize corresponding to the heuristic\\
    - nx1 **Array{Float64,2}** or **Void** expsmoothstepsize: exact stepsize corresponding to the exact expected smoothness constant\\
"""
function get_stepsize_bounds(prob::Prob, simplebound::Array{Float64}, bernsteinbound::Array{Float64}, heuristicbound::Array{Float64}, expsmoothcst)
    n = prob.numdata;
    mu = get_mu_str_conv(prob); # WARNING: this should not be recomputed every time
    Li_s = get_Li(prob);
    Lmax = maximum(Li_s);

    rho_over_n = ( n - (1:n) ) ./ ( (1:n).*(n-1) ); # Sketch residual divided by n
    rightterm = rho_over_n*Lmax + (mu*n)/(4*(1:n)); # Right-hand side term in the max
    println("rightterm = ", rightterm);
    if typeof(expsmoothcst)==Array{Float64,2}
        expsmoothstepsize = 0.25 .* (1 ./ max.(expsmoothcst, rightterm) );
    else
        expsmoothstepsize = nothing;
    end
    simplestepsize = 0.25 .* (1 ./ max.(simplebound, rightterm) );
    bernsteinstepsize = 0.25 .* (1 ./ max.(bernsteinbound, rightterm) );
    heuristicstepsize = 0.25 .* (1 ./ max.(heuristicbound, rightterm) );

    return simplestepsize, bernsteinstepsize, heuristicstepsize, expsmoothstepsize
end

# """
#     save_SAGA_nice_constants()

# Save upper bounds of the expected smoothness constant, 

# #INPUTS:\\
#     - 
# #OUTPUTS:\\
#     - 
# """
# function save_SAGA_nice_constants(prob, data, n, d, mu, L, Lmax, Lbar, Li_s, expsmoothcst, tautheory, tauheuristic)
#     probname = replace(replace(prob.name, r"[\/]", "-"), ".", "_");
#     default_path = "./data/";

#     savename = "-constants";

#     if data in ["gaussian", "diagonal", "lone_eig_val"]
#         savename = string(savename, "-srand_1")
#     end

#     if typeof(expsmoothcst)==Array{Float64,2}
#         save("$(default_path)$(savename).jld", "n", n, "d", d, "mu", mu,
#              "L", L, "Lmax", Lmax, "Lbar", Lbar, "Li_s", Li_s,  "expsmoothcst", expsmoothcst,
#              "tautheory", tautheory, "tauheuristic", tauheuristic);
#     else
#         save("$(default_path)$(savename).jld", "n", n, "d", d, "mu", mu, "L", L, "Lmax", Lmax, "Lbar", Lbar, "Li_s", Li_s);
#     end
# end