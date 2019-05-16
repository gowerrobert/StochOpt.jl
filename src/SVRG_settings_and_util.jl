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
#     optimal_minibatch_L_SVRG_D_nice(prob::Prob, method, options::MyOptions)

# Compute the optimal mini-batch size when the update probability is p = 1/n for the L-SVRG-D algorithm with b-nice sampling.

# # INPUTS
# - **Int64** n: number of data samples
# - **Float64** mu: strong convexity parameter of the objective function
# - **Float64** L: smoothness constant of the whole objective function f
# - **Float64** Lmax: max of the smoothness constant of the f_i functions
# # OUTPUTS
# - **Int64** minibatch_size: theoretical mini-batch size for Free-SVRG with b-nice sampling
# """
# function optimal_minibatch_L_SVRG_D_nice(n, mu, L, Lmax)
#

#     return minibatch_size
# end


"""
    compute_skip_error_SVRG(n::Int64, minibatch_size::Int64, skip_multiplier::Float64=0.02)

Compute the number of skipped iterations between two error estimation.
The computation rule is arbitrary, but depends on the dimension of the problem and on the mini-batch size.

#INPUTS:\\
    - **Int64** n: number of data samples\\
    - **Int64** minibatch\\_size: size of the mini-batch\\
    - **Float64** skip\\_multiplier: arbitrary multiplier\\
#OUTPUTS:\\
    - **Int64** skipped_errors: number iterations between two evaluations of the error\\
"""
function compute_skip_error_SVRG(n::Int64, minibatch_size::Int64, skip_multiplier::Float64=0.015)
    tmp = floor(skip_multiplier*n/minibatch_size)
    skipped_errors = 1
    while(tmp > 1.0)
        tmp /= 2
        skipped_errors *= 2^1
    end
    skipped_errors = convert(Int64, skipped_errors)

    return skipped_errors
end


"""
    simulate_Free_SVRG_nice(prob, minibatchgrid, options,
                       numsimu=1, skipped_errors=1, skip_multiplier=0.02, path="./")

Runs several times (numsimu) mini-batch Free-SVRG with nice sampling for each
mini-batch size in the give list (minibatchgrid) in order to evaluate the
correpsonding average iteration complexity.

#INPUTS:\\
    - **Prob** prob: considered problem, e.g., logistic regression, ridge regression...\\
    - **Array{Int64,1}** minibatchgrid: list of the different mini-batch sizes\\
    - **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...\\
    - **Int64** numsimu: number of runs of mini-batch Free-SVRG\\
    - **Int64** skipped\\_errors: number iterations between two evaluations of the error (-1 for automatic computation)\\
    - **Float64** skip\\_multiplier: multiplier used to compute automatically "skipped_error" (between 0 and 1)\\
    - **AbstractString** path: path to the folder where the plots are saved\\
#OUTPUTS:\\
    - OUTPUTS: output of each run, size length(minibatchgrid)*numsimu\\
    - **Array{Float64,1}** itercomplex: average iteration complexity for each of the mini-batch size over numsimu samples
"""
function simulate_Free_SVRG_nice(prob::Prob, minibatchgrid::Array{Int64,1}, options::MyOptions ;
                                 numsimu::Int64=1, skipped_errors::Int64=-1, skip_multiplier::Float64=0.02, path::AbstractString="./")
    ## Remarks
    ## - One could set skipped_errors inside the loop with skipped_errors = skipped_errors_base/b

    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
    path = string(path, "data/")

    n = prob.numdata

    itercomplex = zeros(length(minibatchgrid), 1) # List of saved outputs
    OUTPUTS = []
    for idx_b in 1:length(minibatchgrid)
        b = minibatchgrid[idx_b]
        println("\nCurrent mini-batch size: ", b)

        if skipped_errors < -1 || skipped_errors == 0
            error("skipped_errors has to be set to -1 (auto) or to a positive integer")
        elseif skipped_errors == -1
            options.skip_error_calculation = compute_skip_error_SVRG(n, b, skip_multiplier)
            println("The number of skipped calculations of the error has been automatically set to ", options.skip_error_calculation)
        else
            options.skip_error_calculation = skipped_errors
        end

        println("---------------------------------- MINI-BATCH ------------------------------------------")
        println(b)
        println("----------------------------------------------------------------------------------------")

        println("---------------------------------- SKIP_ERROR ------------------------------------------")
        println(options.skip_error_calculation)
        println("----------------------------------------------------------------------------------------")

        n = prob.numdata
        mu = prob.mu
        L = prob.L
        Lmax = prob.Lmax

        options.batchsize = b
        for i=1:numsimu
            println("----- Simulation #", i, " -----")
            ## Free-SVRG with optimal b-nice sampling ( m = n, b = b_grid, step size = gamma^*(b) )
            numinneriters = n                                                           # inner loop size set to the number of data points
            options.stepsize_multiplier = -1.0                                          # theoretical step size set in boot_Free_SVRG
            sampling = build_sampling("nice", n, options)
            free = initiate_Free_SVRG(prob, options, sampling, numinneriters=numinneriters, averaged_reference_point=true)

            ## Step size computed externally for monitoring
            stepsize = 1.0/(2.0*(free.expected_smoothness + 2.0*free.expected_residual))
            println("----------------------------- PRACTICAL STEP SIZE --------------------------------------")
            println(stepsize)
            println("----------------------------------------------------------------------------------------")

            output = minimizeFunc(prob, free, options)
            println("Output fail = ", output.fail, "\n")
            itercomplex[idx_b] += output.iterations
            # str_m_free = @sprintf "%d" free.numinneriters
            # str_b_free = @sprintf "%d" free.batchsize
            # str_step_free = @sprintf "%.2e" free.stepsize
            # output.name = latexstring("\$m = n = $str_m_free, b = $str_b_free, \\gamma^*(b) = $str_step_free\$")

            output.name = string("b=", b)
            OUTPUTS = [OUTPUTS; output]
        end
    end

    ## Averaging the last iteration number
    itercomplex = itercomplex ./ numsimu
    itercomplex = itercomplex[:]

    ## Saving the result of the simulations
    savename = "-exp1a-empcomplex-$(numsimu)-avg"
    save("$(path)$(probname)$(savename).jld", "itercomplex", itercomplex, "OUTPUTS", OUTPUTS)

    return OUTPUTS, itercomplex
end


"""
    plot_empirical_complexity_SVRG(prob::Prob, minibatchgrid::Array{Int64,1}, empcomplex::Array{Float64,1},
                              b_practical::Int64, b_empirical::Int64, save_path::AbstractString ;
                              numsimu=1, skip_multiplier=0.0, legendpos=:best)

Saves the plot of the empirical total complexity vs the minibatch size or the inner loop size.

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)\\
    - **Array{Int64,1}** minibatchgrid: list of the different mini-batch sizes\\
    - **Array{Float64,1}** empcomplex: average total complexity (tau*iteration complexity)
      for each of the mini-batch size (tau) over numsimu samples\\
    - **Int64** b_practical: heuristic optimal mini-batch size\\
    - **Int64** b_empirical: empirical optimal mini-batch size\\
    - **AbstractString** save_path: path to the experiment directory\\
    - **Int64** numsimu: number of simulations on which the total complexity is average\\
    - **Symbol** legendpos: position of the legend
#OUTPUTS:\\
    - None
"""
function plot_empirical_complexity_SVRG(prob::Prob, minibatchgrid::Array{Int64,1}, empcomplex::Array{Float64,1}, b_practical::Int64, b_empirical::Int64, save_path::AbstractString ;                                                     numsimu::Int64=1, skip_multiplier::Float64=0.0, legendpos::Symbol=:best)
    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")

    fontmed = 12
    fontbig = 15
    xlabeltxt = "mini-batch size"
    ylabeltxt = "empirical total complexity"

    n = prob.numdata
    d = prob.numfeatures

    labellist = [latexstring("\$b_\\mathrm{empirical} = $b_empirical\$"),
                 latexstring("\$b_\\mathrm{practical} \\; = $b_practical\$")]

    plot(minibatchgrid, empcomplex, linestyle=:solid, color=:black,
         xaxis=:log, yaxis=:log,
         xlabel=xlabeltxt, ylabel=ylabeltxt, label="",
         xticks=(minibatchgrid, minibatchgrid),
         xrotation = 45,
         tickfont=font(fontmed),
         guidefont=font(fontbig), linewidth=3, grid=false)
        #  title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)))
    vline!([b_empirical], line=(:dash, 3), color=:blue, label=labellist[1],
           legendfont=font(fontbig), legend=legendpos) #:legend
    #legendtitle="Optimal mini-batch size")
    vline!([b_practical], line=(:dot, 3), color=:red, label=labellist[2])

    savename = "-exp1a-empcomplex-$(numsimu)-avg"
    if skip_multiplier > 0.0
        savename = string(savename, "_skip_mult_", replace(string(skip_multiplier), "." => "_")); # Extra suffix to check which skip values to keep
    end
    savefig("$(save_path)figures/$(probname)$(savename).pdf")
end