function b_tilde_tight(n::Int, L, Lmax, mu)
    # Tight formula?
    # \tilde{b} &\eqdef & \tfrac{n(3L_{\max} - L)}{n(n-1)\mu - nL + 3L_{\max})}
    # return max(floor(Int, n*(3*Lmax-L)/(n*(n-1)*mu - n*L + 3*Lmax)), 1)
    return n*(3*Lmax-L)/(n*(n-1)*mu - n*L + 3*Lmax)
end
function b_hat_tight(n::Int, L, Lmax)
    # Tight formula?
    # \hat{b} &\eqdef & \sqrt{\tfrac{n}{2}\tfrac{3L_{\max} - L}{nL - 3L_{\max}}}\\
    # return max(floor(Int, sqrt( n*(3*Lmax - L)/(2*(n*L - 3*Lmax)) )), 1)
    return sqrt( n*(3*Lmax - L)/(2*(n*L - 3*Lmax)) )
end

"""
    optimal_minibatch_Free_SVRG_nice_tight(m, n, mu, L, Lmax)

Compute the optimal mini-batch size when the inner loop size m = n for the Free-SVRG algorithm with b-nice sampling,
    based on the theory that takes sketch residual into acount.

# INPUTS
- **Int64** m: inner loop size
- **Int64** n: number of data samples
- **Float64** mu: strong convexity parameter of the objective function
- **Float64** L: smoothness constant of the whole objective function f
- **Float64** Lmax: max of the smoothness constant of the f_i functions
# OUTPUTS
- **Int64** minibatch_size: theoretical mini-batch size for Free-SVRG with b-nice sampling
"""

function optimal_minibatch_Free_SVRG_nice_tight(m, n, mu, L, Lmax)
    if m == n
        flag = "none"
        if n < L/mu
            if 3*Lmax > n*L
                minibatch_size = n
                flag = "n"
            else
                minibatch_size = max(floor(Int, b_hat_tight(n, L, Lmax)), 1)
                flag = "b_hat_tight"
            end
        elseif L/mu <= n <= 3*Lmax/mu
            b_tilde_value = max(floor(Int, b_tilde_tight(n, L, Lmax, mu)), 1) #( 3*n*(Lmax-L) ) / ( mu*n*(n-1) - 3*(n*L-Lmax) )
            if 3*Lmax > n*L
                minibatch_size = b_tilde_value
                flag = "b_tilde_tight"
            else
                b_hat_value = max(floor(Int, b_hat_tight(n, L, Lmax)), 1) #sqrt( ( n*(Lmax-L) ) / ( 2*(n*L-Lmax) ) )
                minibatch_size = floor(Int, min(b_hat_value, b_tilde_value))
                flag = "min_tight"
            end
        else
            minibatch_size = 1
            flag = "1"
        end
        println(flag)
    else
        error("No optimal mini-bath size available for Free-SVRG with this inner loop size")
    end

    return minibatch_size
end

function b_tilde(n::Int, L, Lmax, mu)
    # Previous formula?
    # \tilde{b} &\eqdef & \tfrac{3n(L_{\max} - L)}{n(n-1)\mu- 3(nL - L_{\max})}
    # return max(floor(Int, 3*n*(Lmax-L)/( n*(n-1)*mu- 3*(n*L - Lmax) )), 1)
    return 3*n*(Lmax-L)/( n*(n-1)*mu - 3*(n*L - Lmax) )
end
function b_hat(n::Int, L, Lmax)
    # Previous formula?
    # \hat{b} &\eqdef & \sqrt{\tfrac{n}{2}\tfrac{L_{\max} - L}{nL - L_{\max}}}\\
    # return max(floor(Int, sqrt((n/2)*((Lmax - L)/(n*L - Lmax)))), 1)
    return sqrt((n/2)*((Lmax - L)/(n*L - Lmax)))
end

"""
    optimal_minibatch_Free_SVRG_nice(m, n, mu, L, Lmax)

Compute the optimal mini-batch size when the inner loop size m = n for the Free-SVRG algorithm with b-nice sampling.

# INPUTS
- **Int64** m: inner loop size
- **Int64** n: number of data samples
- **Float64** mu: strong convexity parameter of the objective function
- **Float64** L: smoothness constant of the whole objective function f
- **Float64** Lmax: max of the smoothness constant of the f_i functions
# OUTPUTS
- **Int64** minibatch_size: theoretical mini-batch size for Free-SVRG with b-nice sampling
"""

function optimal_minibatch_Free_SVRG_nice(m, n, mu, L, Lmax)
    if m == n
        flag = "none"
        if n < 3*L/mu
            minibatch_size = n
            flag = "n"
        elseif 3*L/mu <= n <= 3*Lmax/mu
            b_hat_value = max(floor(Int, b_hat(n, L, Lmax)), 1) #sqrt( ( n*(Lmax-L) ) / ( 2*(n*L-Lmax) ) )
            b_tilde_value = max(floor(Int, b_tilde(n, L, Lmax, mu)), 1) # ( 3*n*(Lmax-L) ) / ( mu*n*(n-1) - 3*(n*L-Lmax) )
            minibatch_size = floor(Int, min(b_hat_value, b_tilde_value))
            flag = "min"
        else
            minibatch_size = 1
            flag = "1"
        end
        println(flag)
    else
        error("No optimal mini-bath size available for Free-SVRG with this inner loop size")
    end

    return minibatch_size
end

"""
    optimal_minibatch_L_SVRG_D_nice(p, n, mu, L, Lmax)

Compute the optimal mini-batch size when the update probability is p = 1/n for the L-SVRG-D algorithm with b-nice sampling.

# INPUTS
- **Float64** p: update probability
- **Int64** n: number of data samples
- **Float64** mu: strong convexity parameter of the objective function
- **Float64** L: smoothness constant of the whole objective function f
- **Float64** Lmax: max of the smoothness constant of the f_i functions
# OUTPUTS
- **Int64** minibatch_size: theoretical mini-batch size for Free-SVRG with b-nice sampling
"""
function optimal_minibatch_L_SVRG_D_nice(p, n, mu, L, Lmax)
    if norm(p - 1/n) < 1e-7
        flag = "none"
        ksi_p = ( (7-4*p)*(1-(1-p)^1.5) ) / ( p*(2-p)*(3-2*p) ) # Defined in Theorem 5.1
        if n < 1.5*ksi_p*L/mu
            minibatch_size = n
            flag = "n"
        elseif 1.5*ksi_p*L/mu <= n <= 1.5*ksi_p*Lmax/mu
            b_hat = sqrt( ( n*(Lmax-L) ) / ( 2*(n*L-Lmax) ) )
            b_tilde = ( 1.5*ksi_p*n*(Lmax-L) ) / ( mu*n*(n-1) - 1.5*ksi_p*(n*L-Lmax) )
            if b_tilde < 1
                error("b_tilde is non positive")
            elseif b_hat < 1
                error("b_hat is non positive")
            end
            minibatch_size = max(floor(Int, min(b_hat, b_tilde)), 1)
            flag = "min"
        else
            minibatch_size = 1
            flag = "1"
        end
        println(flag)
    else
        error("No optimal mini-bath size available for L-SVRG-D with this probability")
    end

    return minibatch_size
end


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
    simulate_Free_SVRG_nice(prob, minibatchgrid, options, save_path,
                       numsimu=1, skipped_errors=1, skip_multiplier=0.02)

Runs several times (numsimu) mini-batch Free-SVRG with nice sampling for each
mini-batch size in the given list (minibatchgrid) in order to evaluate the
corresponding average iteration complexity.

#INPUTS:\\
    - **Prob** prob: considered problem, e.g., logistic regression, ridge regression...\\
    - **Array{Int64,1}** minibatchgrid: list of the different mini-batch sizes\\
    - **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...\\
    - **AbstractString** save_path: path to the folder where the plots are saved\\
    - **Int64** numsimu: number of runs of mini-batch Free-SVRG\\
    - **Int64** skipped\\_errors: number iterations between two evaluations of the error (-1 for automatic computation)\\
    - **Float64** skip\\_multiplier: multiplier used to compute automatically "skipped_error" (between 0 and 1)\\
    - **AbstractString** suffix: suffix added to saved file names\\
#OUTPUTS:\\
    - OUTPUTS: output of each run, size length(minibatchgrid)*numsimu\\
    - **Array{Float64,1}** itercomplex: average iteration complexity for each of the mini-batch sizes over numsimu samples
"""
function simulate_Free_SVRG_nice(prob::Prob, minibatchgrid::Array{Int64,1}, options::MyOptions, save_path::AbstractString ;
                                 numsimu::Int64=1, skipped_errors::Int64=-1, skip_multiplier::Float64=0.02, suffix::AbstractString="")
    ## Remarks
    ## - One could set skipped_errors inside the loop with skipped_errors = skipped_errors_base/b

    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")

    if suffix != ""
        suffix = "-$(suffix)"
    end

    n = prob.numdata
    mu = prob.mu
    L = prob.L
    Lmax = prob.Lmax

    itercomplex = zeros(length(minibatchgrid), 1) # List of saved outputs
    OUTPUTS = []
    for idx_b in 1:length(minibatchgrid)
        b = minibatchgrid[idx_b]
        println("\nCurrent mini-batch size: ", b)

        if skipped_errors < -1 || skipped_errors == 0
            error("skipped_errors has to be set to -1 (auto) or to a positive integer")
        elseif skipped_errors == -1
            options.skip_error_calculation = compute_skip_error_SVRG(n, b, skip_multiplier) ## Maybe a better parametrization exists ...
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

        options.batchsize = b
        for i=1:numsimu
            println("----- Simulation #", i, " -----")
            ################################################################################
            ############################## OPTIMAL FREE-SVRG ###############################
            ################################################################################
            ## Free-SVRG with optimal b-nice sampling ( m = n, b = b_grid, step size = gamma^*(b) )
            numinneriters = n                              # inner loop size set to the number of data points
            options.stepsize_multiplier = -1.0             # theoretical step size set in boot_Free_SVRG
            sampling = build_sampling("nice", n, options)
            free = initiate_Free_SVRG(prob, options, sampling, numinneriters=numinneriters, averaged_reference_point=true)
            output = minimizeFunc(prob, free, options)

            println("----------------------------- # EPOCHS[END] -------------------------------------")
            println(output.epochs[end])
            println("---------------------------------------------------------------------------------")

            println("Output fail = ", output.fail, "\n")
            itercomplex[idx_b] += output.iterations

            output.name = string("b=", b)
            OUTPUTS = [OUTPUTS; output]
        end
    end

    ## Averaging the iteration complexity
    itercomplex = itercomplex ./ numsimu
    itercomplex = itercomplex[:]

    ## Saving the result of the simulations
    savename = "-exp3-empcomplex-$(numsimu)-avg$(suffix)"
    save("$(save_path)data/$(probname)$(savename).jld", "itercomplex", itercomplex, "OUTPUTS", OUTPUTS)

    return OUTPUTS, itercomplex
end

"""
    simulate_L_SVRG_D_nice(prob, minibatchgrid, options,
                       numsimu=1, skipped_errors=1, skip_multiplier=0.02, path="./")

Runs several times (numsimu) mini-batch Loopless-SVRG-Decreasing with nice sampling for each
mini-batch size in the given list (minibatchgrid) in order to evaluate the
corresponding average total complexity.

#INPUTS:\\
    - **Prob** prob: considered problem, e.g., logistic regression, ridge regression...\\
    - **Array{Int64,1}** minibatchgrid: list of the different mini-batch sizes\\
    - **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...\\
    - **Int64** numsimu: number of runs of mini-batch L-SVRG-D\\
    - **Int64** skipped\\_errors: number iterations between two evaluations of the error (-1 for automatic computation)\\
    - **Float64** skip\\_multiplier: multiplier used to compute automatically "skipped_error" (between 0 and 1)\\
    - **AbstractString** path: path to the folder where the plots are saved\\
#OUTPUTS:\\
    - OUTPUTS: output of each run, size length(minibatchgrid)*numsimu\\
"""
function simulate_L_SVRG_D_nice(prob::Prob, minibatchgrid::Array{Int64,1}, options::MyOptions ;
                                 numsimu::Int64=1, skipped_errors::Int64=-1, skip_multiplier::Float64=0.02, path::AbstractString="./")

    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
    path = string(path, "data/")

    n = prob.numdata

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
            ################################################################################
            ############################### OPTIMAL L-SVRG-D ###############################
            ################################################################################
            ## L_SVRG_D with 1-nice sampling ( p = 1/n, b = 1, step size = gamma^*(b) )
            proba = 1/n                         # update probability set to the inverse of the number of data points
            options.stepsize_multiplier = -1.0  # theoretical step size set in boot_L_SVRG_D
            sampling = build_sampling("nice", n, options)
            decreasing = initiate_L_SVRG_D(prob, options, sampling, proba)
            output = minimizeFunc(prob, decreasing, options)

            println("----------------------------- # EPOCHS[END] -------------------------------------")
            println(output.epochs[end])
            println("----------------------------------------------------------------------------------------")

            println("Output fail = ", output.fail, "\n")

            output.name = string("b=", b)
            OUTPUTS = [OUTPUTS; output]
        end
    end

    ## Saving the result of the simulations
    savename = "-exp1a-decreasing-empcomplex-$(numsimu)-avg"
    save("$(path)$(probname)$(savename).jld", "OUTPUTS", OUTPUTS)

    return OUTPUTS
end


"""
    simulate_Free_SVRG_nice_inner_loop(prob, inner_loop_grid, options,
                       numsimu=1, skipped_errors=1, skip_multiplier=0.02, path="./")

Runs several times (numsimu) Free-SVRG with 1-nice sampling for each
inner loop size m in the given list (inner_loop_grid) in order to evaluate the
correpsonding average iteration complexity.

#INPUTS:\\
    - **Prob** prob: considered problem, e.g., logistic regression, ridge regression...\\
    - **Array{Int64,1}** inner_loop_grid: list of the different inner loop sizes\\
    - **MyOptions** options: different options such as the mini-batch size, the stepsize multiplier...\\
    - **Int64** numsimu: number of runs of 1-nice Free-SVRG\\
    - **Int64** skipped\\_errors: number iterations between two evaluations of the error (-1 for automatic computation)\\
    - **Float64** skip\\_multiplier: multiplier used to compute automatically "skipped_error" (between 0 and 1)\\
    - **AbstractString** path: path to the folder where the plots are saved\\
#OUTPUTS:\\
    - OUTPUTS: output of each run, size length(inner_loop_grid)*numsimu\\
    - **Array{Float64,1}** itercomplex: average iter complexity for each of the inner loop sizes over numsimu samples
"""
function simulate_Free_SVRG_nice_inner_loop(prob::Prob, inner_loop_grid::Array{Int64,1}, options::MyOptions ;
                                           numsimu::Int64=1, skipped_errors::Int64=-1, skip_multiplier::Float64=0.02, path::AbstractString="./")
    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_");
    path = string(path, "data/")

    n = prob.numdata

    if skipped_errors < -1 || skipped_errors == 0
        error("skipped_errors has to be set to -1 (auto) or to a positive integer")
    elseif skipped_errors == -1
        options.skip_error_calculation = compute_skip_error_SVRG(n, 1, skip_multiplier)
        println("The number of skipped calculations of the error has been automatically set to ", options.skip_error_calculation)
    else
        options.skip_error_calculation = skipped_errors
    end

    n = prob.numdata
    mu = prob.mu
    L = prob.L
    Lmax = prob.Lmax

    itercomplex = zeros(length(inner_loop_grid), 1) # List of saved outputs
    OUTPUTS = []
    for idx_m in 1:length(inner_loop_grid)
        m = inner_loop_grid[idx_m]
        println("\nCurrent inner loop size: ", m)

        println("-------------------------------- INNER LOOP SIZE ---------------------------------------")
        println(m)
        println("----------------------------------------------------------------------------------------")

        println("---------------------------------- SKIP_ERROR ------------------------------------------")
        println(options.skip_error_calculation)
        println("----------------------------------------------------------------------------------------")

        for i=1:numsimu
            println("----- Simulation #", i, " -----")
            ################################################################################
            ############################### 1-NICE FREE-SVRG ###############################
            ################################################################################
            ## Free-SVRG with 1-nice sampling ( m = n, b = 1, step size = gamma^*(b) )
            options.batchsize = 1                          # 1-nice sampling
            numinneriters = m                              # inner loop size set to the number of data points
            options.stepsize_multiplier = -1.0             # theoretical step size set in boot_Free_SVRG
            sampling = build_sampling("nice", n, options)
            free = initiate_Free_SVRG(prob, options, sampling, numinneriters=numinneriters, averaged_reference_point=true)
            output = minimizeFunc(prob, free, options)

            println("----------------------------- # EPOCHS[END] -------------------------------------")
            println(output.epochs[end])
            println("---------------------------------------------------------------------------------")

            println("Output fail = ", output.fail, "\n")
            itercomplex[idx_m] += output.iterations

            output.name = string("m=", m)
            OUTPUTS = [OUTPUTS; output]
        end
    end

    ## Averaging the iteration complexity
    itercomplex = itercomplex ./ numsimu
    itercomplex = itercomplex[:]

    ## Saving the result of the simulations
    savename = "-exp4-empcomplex-$(numsimu)-avg$(suffix)"
    save("$(save_path)data/$(probname)$(savename).jld", "itercomplex", itercomplex, "OUTPUTS", OUTPUTS)

    return OUTPUTS, itercomplex
end


"""
    plot_empirical_complexity_SVRG(prob, exp_number, grid, empcomplex, b_practical, b_empirical, save_path ;
                              numsimu=1, skip_multiplier=0.0, legendpos=:best, suffix="")

Saves the plot of the empirical total complexity vs the mini-batch or the inner loop size.

#INPUTS:\\
    - **Prob** prob: considered problem, i.e. logistic regression, ridge ression... (see src/StochOpt.jl)\\
    - **Int64** exp_number: number of the experiment:
        - 1 for mini-batch optimality of Free-SVRG
        - 2 for inner loop size optimality of Free-SVRG
        - 3 for mini-batch optimality of L-SVRG-D
    - **Array{Int64,1}** grid: list of the different mini-batch or inner loop sizes\\
    - **Array{Float64,1}** empcomplex: average total complexity for each of point of the grid over numsimu samples\\
    - **Int64** theoretical_value: theoretical optimal value\\
    - **Int64** empirical_value: empirical optimal value\\
    - **AbstractString** save_path: path to the experiment directory\\
    - **Int64** numsimu: number of simulations on which the total complexity is average\\
    - **Symbol** legendpos: position of the legend\\
    - **AbstractString** suffix: suffix to append to the output file name
#OUTPUTS:\\
    - None
"""
function plot_empirical_complexity_SVRG(prob::Prob, exp_number::Int64, grid::Array{Int64,1}, empcomplex::Array{Float64,1}, theoretical_value::Int64, empirical_value::Int64, save_path::AbstractString ; numsimu::Int64=1, skip_multiplier::Float64=0.0, legendpos::Symbol=:best, suffix::AbstractString="")
    if !(exp_number in 1:3)
        error("Wrong choice of exp_number")
    end

    probname = replace(replace(prob.name, r"[\/]" => "-"), "." => "_")

    fontmed = 12
    fontbig = 15
    ylabeltxt = "empirical total complexity"

    n = prob.numdata
    d = prob.numfeatures

    if exp_number == 1
        xlabeltxt = "mini-batch size"
        labellist = [latexstring("\$b_\\mathrm{empirical} = $empirical_value\$"),
                     latexstring("\$b_\\mathrm{theory} \\; = $theoretical_value\$")]
        savename = "-exp1a-empcomplex-$(numsimu)-avg"
    elseif exp_number == 2
        xlabeltxt = "inner loop size"
        labellist = [latexstring("\$m_\\mathrm{empirical} = $empirical_value\$"),
                     latexstring("\$m_\\mathrm{theory} \\; = $theoretical_value\$")]
        savename = "-exp1b-empcomplex-$(numsimu)-avg"
    elseif exp_number == 3
        xlabeltxt = "mini-batch size"
        labellist = [latexstring("\$b_\\mathrm{empirical} = $empirical_value\$"),
                     latexstring("\$b_\\mathrm{theory} \\; = $theoretical_value\$")]
        savename = "-exp1a-decreasing-empcomplex-$(numsimu)-avg"
    end

    plot(grid, empcomplex, linestyle=:solid, color=:black,
         xaxis=:log, yaxis=:log,
         xlabel=xlabeltxt, ylabel=ylabeltxt, label="",
         xticks=(grid, grid),
         xrotation = 45,
         tickfont=font(fontmed),
         guidefont=font(fontbig), linewidth=3, grid=false)
        #  title=string("Pb: ", probname, ", n=", string(n), ", d=", string(d)))

    vline!([empirical_value], line=(:dash, 3), color=:blue, label=labellist[1],
           legendfont=font(fontbig), legend=legendpos) #:legend
    #legendtitle="Optimal mini-batch size")
    vline!([theoretical_value], line=(:dot, 3), color=:red, label=labellist[2])

    if skip_multiplier > 0.0
        savename = string(savename, "_skip_mult_", replace(string(skip_multiplier), "." => "_")) # Extra suffix to check which skip values to keep
        savename = string(savename, suffix)
    end
    savefig("$(save_path)figures/$(probname)$(savename).pdf")
end