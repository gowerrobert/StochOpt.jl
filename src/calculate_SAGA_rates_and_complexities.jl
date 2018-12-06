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
    # Matrix(prob.X[:, C]'*prob.X[:, C])
    # println("Symmetric")
    # Symmetric(Matrix(prob.X[:, C]'*prob.X[:, C]))
    # println("Eigmax")
    # eigmax(Symmetric(Matrix(prob.X[:, C]'*prob.X[:, C])))
    LC = 0;
    if(length(C) < prob.numfeatures)
        try
            LC = eigmax(Symmetric(Matrix(prob.X[:, C]'*prob.X[:, C])))/length(C) + prob.lambda;
        catch loaderror # Uses power iteration if eigmax fails
            # println("Using power iteration instead of eigmax which returns the following error: ", loaderror);
            LC = power_iteration(Symmetric(Matrix(prob.X[:, C]'*prob.X[:, C])))/length(C) + prob.lambda;
        end
    else
        try
            LC = eigmax(Symmetric(Matrix(prob.X[:,C]*prob.X[:,C]')))/length(C) + prob.lambda;
        catch loaderror # Uses power iteration if eigmax fails
            # println("Using power iteration instead of eigmax which returns the following error: ", loaderror);
            LC = power_iteration(Symmetric(Matrix(prob.X[:, C]*prob.X[:, C]')))/length(C) + prob.lambda;
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
        Ls[C] = Ls[C] .+ (1/c1)*get_LC(prob, C); # Implementation without inner loop
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
        mu = eigmin(Matrix(prob.X*prob.X'))/prob.numdata + prob.lambda; # julia 0.7 'full(A)' has been deprecated
    else
        mu = eigmin(Matrix(prob.X'*prob.X))/prob.numdata + prob.lambda; # julia 0.7 'full(A)' has been deprecated
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
a heuristic estimation of it and its exact value (if there are few data points) for 
each mini-batch size ``τ`` from 1 to n.

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge regression...
      (see src/StochOpt.jl)\\
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
        if(n<100)
            print("Calculating bounds for tau = ", tau, "\n");
        elseif((tau % floor(Int64, (n/100))) == 1) # 100 messages for the whole data set
            print("Calculating bounds for tau = ", tau, "\n");
        end
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

Compute upper bounds of the stepsize based on the simple bound, the Bernstein bound, 
our heuristic and the exact expected smoothness constant for each mini-batch size 
``τ`` from 1 to n.

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge regression... 
      (see src/StochOpt.jl)\\
    - nx1 **Array{Float64,2}** simplebound: simple upper bound\\
    - nx1 **Array{Float64,2}** bernsteinbound: Bernstein upper bound\\
    - nx1 **Array{Float64,2}** heuristicbound: heuristic estimation\\
    - nx1 **Array{Float64,2}** or **Void** expsmoothcst: exact expected smoothness constant\\
#OUTPUTS:\\
    - nx1 **Array{Float64,2}** simplestepsize: lower bound of the stepsize corresponding to 
      the simple upper bound\\
    - nx1 **Array{Float64,2}** bernsteinstepsize: lower bound of the stepsize corresponding to 
      the Bernstein upper bound\\
    - nx1 **Array{Float64,2}** heuristicstepsize: lower bound of the stepsize corresponding to 
      the heuristic\\
    - nx1 **Array{Float64,2}** or **Void** expsmoothstepsize: exact stepsize corresponding to 
      the exact expected smoothness constant\\
"""
function get_stepsize_bounds(prob::Prob, simplebound::Array{Float64}, 
                             bernsteinbound::Array{Float64}, heuristicbound::Array{Float64}, 
                             expsmoothcst)
    n = prob.numdata;
    mu = get_mu_str_conv(prob); # WARNING: this should not be recomputed every time
    Li_s = get_Li(prob);
    Lmax = maximum(Li_s);

    rho_over_n = ( n .- (1:n) ) ./ ( (1:n).*(n-1) ); # Sketch residual divided by n
    rightterm = rho_over_n*Lmax + ((mu*n)/(4*(1:n)))'; # Right-hand side term in the max
    
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

"""
    save_SAGA_nice_constants(prob::Prob, data::String, 
                             simplebound::Array{Float64}, bernsteinbound::Array{Float64}, 
                             heuristicbound::Array{Float64}, expsmoothcst, 
                             simplestepsize::Array{Float64}, bernsteinstepsize::Array{Float64}, 
                             heuristicstepsize::Array{Float64}, expsmoothstepsize,
                             opt_minibatch_simple::Int64, opt_minibatch_bernstein::Int64, 
                             opt_minibatch_heuristic::Int64, opt_minibatch_exact::Int64)

Saves the problem, caracteristic constants of the problem, the upper bounds of the expected 
smoothness constant, and corresponding estimation of the optimal mini-batch size. 
It also saves the exact expected smoothness constant and its corresponding 
optimal mini-batch size (both are of type Void if not available).

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge regression... 
      (see src/StochOpt.jl)\\
    - String data: selected data (artificially generated or dataset)\\
    - nx1 **Array{Float64,2}** simplebound: simple upper bound\\
    - nx1 **Array{Float64,2}** bernsteinbound: Bernstein upper bound\\
    - nx1 **Array{Float64,2}** heuristicbound: heuristic estimation\\
    - nx1 **Array{Float64,2}** or **Void** expsmoothcst: exact expected smoothness constant\\
    - nx1 **Array{Float64,2}** simplestepsize: lower bound of the stepsize corresponding to 
      the simple upper bound\\
    - nx1 **Array{Float64,2}** bernsteinstepsize: lower bound of the stepsize corresponding to 
      the Bernstein upper bound\\
    - nx1 **Array{Float64,2}** heuristicstepsize: lower bound of the stepsize corresponding to 
      the heuristic\\
    - nx1 **Array{Float64,2}** or **Void** expsmoothstepsize: exact stepsize corresponding to 
      the exact expected smoothness constant\\
    - **Int64** opt\\_minibatch\\_simple: simple bound optimal mini-batch size estimate\\
    - **Int64** opt\\_minibatch\\_bernstein: bernstein bound optimal mini-batch size estimate\\
    - **Int64** opt\\_minibatch\\_heuristic: heuristic optimal mini-batch size estimate\\
    - **Int64** opt\\_minibatch\\_exact: theoretical exact optimal mini-batch size\\
#OUTPUTS:\\
    - None
"""
function save_SAGA_nice_constants(prob::Prob, data::String, 
                                  simplebound::Array{Float64}, bernsteinbound::Array{Float64}, 
                                  heuristicbound::Array{Float64}, expsmoothcst, 
                                  simplestepsize::Array{Float64}, bernsteinstepsize::Array{Float64}, 
                                  heuristicstepsize::Array{Float64}, expsmoothstepsize,
                                  opt_minibatch_simple::Int64, opt_minibatch_bernstein::Int64, 
                                  opt_minibatch_heuristic::Int64, opt_minibatch_exact::Int64)
    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_"); # julia 0.7
    default_path = "./data/";
    savename = "-cst";

    n = prob.numdata;
    d = prob.numfeatures;    

    ## WARNING: this should not be recomputed every time
    mu = get_mu_str_conv(prob); 
    L = get_LC(prob, collect(1:n));
    Li_s = get_Li(prob);
    Lmax = maximum(Li_s);
    Lbar = mean(Li_s);

    ## Saving the seed used to generate data
    if data in ["gaussian", "diagonal", "lone_eig_val"]
        seed = string(reinterpret(Int32, Random.GLOBAL_RNG.seed[1]));
        seed = string("_seed-", seed);
        savename = string(seed, savename);
    end

    ## Saving the problem and the related constants
    save("$(default_path)$(probname)$(savename).jld", 
         "mu", mu, "L", L, "Lmax", Lmax, "Lbar", Lbar, "Li_s", Li_s,
         "simplebound", simplebound, "bernsteinbound", bernsteinbound, 
         "heuristicbound", heuristicbound, "expsmoothcst", expsmoothcst, 
         "simplestepsize", simplestepsize, "bernsteinstepsize", bernsteinstepsize, 
         "heuristicstepsize", heuristicstepsize, "expsmoothstepsize", expsmoothstepsize,
         "opt_minibatch_simple", opt_minibatch_simple, "opt_minibatch_bernstein", opt_minibatch_bernstein, 
         "opt_minibatch_heuristic", opt_minibatch_heuristic, "opt_minibatch_exact", opt_minibatch_exact);
end

"""
    compute_skip_error(n::Int64, minibatch_size::Int64, skip_multiplier::Float64=0.02)

Compute the number of skipped iterations between two error estimation. 
The computation rule is arbitrary, but depends on the dimension of the problem and on the mini-batch size.

#INPUTS:\\
    - **Int64** n: number of data samples\\
    - **Int64** minibatch\\_size: size of the mini-batch\\
    - **Float64** skip\\_multiplier: arbitrary multiplier\\
#OUTPUTS:\\
    - **Int64** skipped_errors: number iterations between two evaluations of the error\\
"""
function compute_skip_error(n::Int64, minibatch_size::Int64, skip_multiplier::Float64=0.015)
    tmp = floor(skip_multiplier*n/minibatch_size);
    skipped_errors = 1;
    while(tmp > 1.0)
        tmp /= 2;
        skipped_errors *= 2;
    end
    skipped_errors = convert(Int64, skipped_errors); # Seems useless

    return skipped_errors
end

"""
    simulate_SAGA_nice(prob, minibatchlist, numsimu=1, 
                       tolerance=10.0^(-3), skipped_errors=1,
                       max_iter=10^8, max_time=10000.0, max_epochs=10000)

Runs several times (numsimu) mini-batch SAGA with nice sampling for each
mini-batch size in the give list (minibatchlist) in order to evaluate the
correpsonding average iteration complexity.

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge regression...
    - **Array{Int64,1}** minibatchlist: list of the different mini-batch sizes\\
    - **Int64** numsimu: number of runs of mini-batch SAGA\\
    - **Float64** tolerance: relative error convergence threshold, at last iteration we have (f(x)-fsol)/(f0-fsol) < epsilon\\
    - **Int64** skipped\\_errors: number iterations between two evaluations of the error (-1 for automatic computation)\\
    - **Float64** skip\\_multiplier: multiplier used to compute automatically "skipped_error" (between 0 and 1)\\
    - **Int64** max\\_iter: maximum number of iterations\\
    - **Float64** max\\_time: maximum run time\\
    - **Int64** max\\_epochs: maximum number of epochs\\
#OUTPUTS:\\
    - OUTPUTS: output of each run, size length(minibatchlist)*numsimu\\
    - **Array{Float64,1}** itercomplex: average iteration complexity for each of the mini-batch size over numsimu samples
"""
function simulate_SAGA_nice(prob::Prob, minibatchlist::Array{Int64,1}, numsimu::Int64 ;
                            tolerance::Float64=10.0^(-3),
                            skipped_errors::Int64=-1, skip_multiplier::Float64=0.02,
                            max_iter::Int64=10^8, max_time::Float64=3600.0, max_epochs::Int64=100)
    ## Remarks
    ## - One could set skipped_errors inside the loop with skipped_errors = skipped_errors_base/tau
    
    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
    default_path = "./data/";

    n = prob.numdata;

    itercomplex = zeros(length(minibatchlist), 1); # List of saved outputs
    OUTPUTS = [];
    # fail = true;
    for idxtau in 1:length(minibatchlist) # 1:n
        tau = minibatchlist[idxtau];
        println("\nCurrent mini-batch size: ", tau);

        if(skipped_errors==-1)
            skipped_errors = compute_skip_error(n, tau, skip_multiplier);
            println("The variable skipped_errors has been automatically set to ", skipped_errors);
        end
        options = set_options(tol=tolerance, max_iter=max_iter, max_time=max_time, max_epocs=max_epochs,
                              initial_point="zeros", # is fixed not to add more randomness
                        #   repeat_stepsize_calculation=true,
                              skip_error_calculation=skipped_errors,
                              force_continue=false); # force continue if diverging or if tolerance reached
        options.batchsize = tau;
        for i=1:numsimu
            println("----- Simulation #", i, " -----");
            # sg = initiate_SAGA(prob, options, minibatch_type="nice"); # Old and diverging implementation
            sg = initiate_SAGA_nice(prob, options); # new separated implementation
            # println("STEPSIZE OF sg: ", sg.stepsize);
            output = minimizeFunc(prob, sg, options);
            println("Output fail = ", output.fail, "\n");
            # fail = !(output.fail == "tol-reached");
            # while(fail)
            #     println("ENTERING THE WHILE LOOP")
            #     sg = initiate_SAGA(prob, options, minibatch_type="nice");
            #     output = minimizeFunc(prob, sg, options);
            #     println("Output fail = ", output.fail);
            #     fail = !(output.fail == "tol-reached");
            # end
            itercomplex[idxtau] += output.iterations;
            output.name = string("\\tau=", tau); # Tiny modification for smaller legend name / Latex symbols in strings
            OUTPUTS = [OUTPUTS; output]; # Array{Any,1} or Array{Output,1}?
        end
    end
    itercomplex = itercomplex ./ numsimu; # simply averaging the last iteration number
    itercomplex = itercomplex[:];

    ## Saving the result of the simulations
    savename = "-empcomplex-$(numsimu)-avg";
    save("$(default_path)$(probname)$(savename).jld", "itercomplex", itercomplex, "OUTPUTS", OUTPUTS);

    return OUTPUTS, itercomplex
end