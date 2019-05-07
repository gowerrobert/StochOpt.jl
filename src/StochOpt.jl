# The types exported
#export MyOptions, Prob, Method, Output
# The exported functions
#export pseudoinvert_matrix, uniform_mat_rank, uniform_sym_rank
#export plot_outputs_Plots

using SparseArrays
using LinearAlgebra
using Printf
using Random
using LaTeXStrings

mutable struct MyOptions
    tol::Float64
    aux::Float64
    max_iter::Int
    skip_error_calculation::Int
    max_time::Float64
    max_epocs::Int64
    printiters::Bool
    exacterror::Bool
    repeat_stepsize_calculation::Bool
    batchsize
    regularizor_parameter::AbstractString
    stepsize_multiplier::Float64
    precondition::Bool
    force_continue::Bool
    rep_number::Int64 # number of times the optimization should be repeated. Only the average is reported.
    embeddim# Hessian_type::AbstractString
    initial_point::AbstractString
end

function set_options(; tol::Float64 = 10.0^(-6.0),
    aux::Float64 = Inf,
    max_iter::Int = 10^8,
    skip_error_calculation::Int = 0,
    max_time::Float64 = 350.0,
    max_epocs::Int64 = 30,
    printiters::Bool = true,
    exacterror::Bool = true,
    repeat_stepsize_calculation::Bool = false,
    batchsize::Int64 = 100,
    regularizor_parameter::AbstractString = "normalized",
    stepsize_multiplier::Float64 = 1.0,
    precondition::Bool = false,
    force_continue::Bool = true,
    rep_number::Int64 = 5, # number of times the optimization should be repeated. Only the average is reported.
    embeddim = 0,
    initial_point = "zeros")

    options = MyOptions(tol, aux, max_iter, skip_error_calculation, max_time, max_epocs,
        printiters, exacterror, repeat_stepsize_calculation, batchsize, "normalized", stepsize_multiplier, precondition, force_continue, rep_number, embeddim, initial_point)
end

mutable struct DataScaling
    rowscaling::Array{Float64}
    colscaling::Array{Float64}
    colsmean::Array{Float64}
    name::AbstractString
end

mutable struct Prob
    X  # why not a sparse array?
    y::Array{Float64}
    numfeatures::Int64
    numdata::Int64
    fsol::Float64
    name::AbstractString
    datascaling::DataScaling
    f_eval::Function
    g_eval::Function
    g_eval!::Function
    Jac_eval!::Function
    scalar_grad_eval::Function #This is phi' in the linear model sum_i phi_i( <x_i,w> -y_i)
    scalar_grad_hess_eval::Function #This is phi' and phi'' in the linear model sum_i phi_i( <x_i,w> -y_i)
    Hess_eval::Function# Calculates the Hessian Hess_eval,Hess_opt,Hess_vv,Hess_D
    Hess_eval!::Function# Calculates the Hessian Hess_eval,Hess_opt,Hess_vv,Hess_D
    Hess_opt::Function # Calculates the Hessian-vector product Hv
    Hess_opt!::Function # Calculates the Hessian-vector product Hv
    Hess_D::Function   # Calculates Diagonal
    Hess_D!::Function   # Calculates Diagonal
    Hess_C::Function   # Gets subset of the columns of Hessian
    Hess_C!::Function   # Gets subset of the columns of Hessian
    Hess_C2::Function
    #    Hess_vv::Function  # Calculates the Hessian-vector-vector product v'Hv
    ## SUGGESTION: add L, Lbar, Lis and L_max as attributes of the problem (instead of attribute of the SAGA method)
    lambda::Float64
    mu::Float64 # Strong-convexity constant
    L::Float64 # Smoothness constant of the whole objective function f
    Lmax::Float64 # Max of the smoothness constant of the f_i functions
    Lbar::Float64 # Average of the smoothness constant of the f_i functions
end

mutable struct SAGAmethod
    epocsperiter::Float64
    gradsperiter::Float64
    name::AbstractString
    stepmethod::Function
    bootmethod::Function
    minibatches::Array{Int64}
    minibatch_type::AbstractString  # Type of mini-batching, e.g, tau-nice, tau-partition, tau-with replacement
    unbiased::Bool        # unbiased = true for SAGA and unbiased = false for SAG
    Jac::Array{Float64}  #Jacobian estimate
    Jacsp::SparseMatrixCSC #Sparse JAcobian estimate
    SAGgrad::Array{Float64}  # The SAG estimate of full gradient, needed for computing unbiased gradient estimate.
    gi::Array{Float64} # Storage for a single stochastic gradient
    aux::Array{Float64}  # Storage for an auxiliary vector, used as the update vectoe
    stepsize::Float64     # The stepsize
    probs::Array{Float64}  # Probability of selecting a coordinate or mini-batch
    probability_type::AbstractString # type of probabilities used, e.g., uniform, nonuniform, nonuniform_opt
    Z    # normalizing variable for probabilities
    L::Float64 # Smoothness constant of the whole objective function f
    Lmax::Float64 # Max of the smoothness constant of the f_i functions
    mu::Float64
end

mutable struct SAGA_nice_method
    epocsperiter::Float64
    gradsperiter::Float64
    name::AbstractString
    stepmethod::Function
    bootmethod::Function
    minibatches::Array{Int64}
    unbiased::Bool        # unbiased = true for SAGA and unbiased = false for SAG
    Jac::Array{Float64}  # Jacobian estimate
    Jacsp::SparseMatrixCSC # Sparse JAcobian estimate
    SAGgrad::Array{Float64}  # The SAG estimate of full gradient, needed for computing unbiased gradient estimate.
    gi::Array{Float64} # Storage for a single stochastic gradient
    aux::Array{Float64}  # Storage for an auxiliary vector, used as the update vector
    stepsize::Float64     # The stepsize
    probs::Array{Float64}  # Probability of selecting a coordinate or mini-batch
    Z    # normalizing variable for probabilities
    reset::Function # reset the parameters of the method like after initiate_SAGA_nice_method
    # L::Float64 # Smoothness constant of the whole objective function f
    # Lmax::Float64 # Max of the smoothness constant of the f_i functions
    # Lbar::Float64 # Average of the smoothness constant of the f_i functions
    # mu::Float64 # Strong-convexity constant
end

mutable struct SVRG_nice_method
    ## SVRG original algorithm with option I
    ## Ref: Accelerating stochastic gradient descent using predictive variance reduction, R. Johnson and T. Zhang, NIPS (2013)
    epocsperiter::Float64
    gradsperiter::Float64
    name::AbstractString
    stepmethod::Function # /!\ mutating function
    bootmethod::Function # /!\ mutating function
    batchsize::Int64
    stepsize::Float64 # step size
    probs::Array{Float64} # probability of selecting a coordinate
    Lmax::Float64 # max of the smoothness constant of the f_i functions
    mu::Float64 # strong-convexity constant
    numinneriters::Int64 # number of inner iterations, usually denoted m
    reference_point::Array{Float64}
    reference_grad::Array{Float64}
    reset::Function # reset some parameters of the method
end

mutable struct free_SVRG_nice_method
    epocsperiter::Float64
    gradsperiter::Float64
    name::AbstractString
    stepmethod::Function # /!\ mutating function
    bootmethod::Function # /!\ mutating function
    batchsize::Int64
    stepsize::Float64 # step size
    probs::Array{Float64} # probability of selecting a coordinate
    Z # normalizing variable for probabilities
    L::Float64 # smoothness constant of the whole objective function f
    Lmax::Float64 # max of the smoothness constant of the f_i functions
    mu::Float64 # strong-convexity constant
    expected_smoothness::Float64 # Expected smoothness constant
    expected_residual::Float64 # expected residual
    numinneriters::Int64 # number of inner iterations, usually denoted m
    reference_point::Array{Float64}
    new_reference_point::Array{Float64} # average on the fly of the inner loop iterates
    reference_grad::Array{Float64}
    averaging_weights::Array{Float64} # averaging weights of the output of the inner loop
    reset::Function # reset some parameters of the method
end

mutable struct SPIN
    epocsperiter::Float64
    gradsperiter::Float64
    name::AbstractString
    stepmethod::Function
    bootmethod::Function
    sketchsize::Int64
    dn::Array{Float64}   # currrent estimate of Newton direction
    rhs::Array{Float64}
    S::Array{Float64}  # Sketched Hessian
    HS::Array{Float64}  # Sketched Hessian
    SHS::Array{Float64}  # Sketched Hessian
    grad::Array{Float64} # gradient
    stepsize::Float64
    sketchtype::AbstractString
end

mutable struct Method
    epocsperiter::Float64
    gradsperiter::Float64
    name::AbstractString
    stepmethod::Function
    grad::Array{Float64}  # Current estimate of full gradient
    gradsamp::Array{Float64} # subsampled local gradient
    S:: Array{Float64}   # embedding space
    H:: Array{Float64}   # An approximation of the  Hessian
    Hsp::SparseMatrixCSC # Sparse Hessian matrix
    HS::Array{Float64}   # action of the  Hessian on S
    HSi::Array{Float64}   # action of the ith Hessian
    SHS:: Array{Float64}   # An embedded Hessian
    stepsize::Float64     # The stepsize
    prevx::Array{Float64} # Storing a reference point or previous point
    diffpnt::Array{Float64} #Stores differences between old and new
    Sold::Array{Float64} #old embedding space
    ind::Array{Int64} # place holder for an array of indices
    aux::Array{Float64} # An auxiliary vector used in computations
    numinneriters::Int64
end

mutable struct Output
    name::AbstractString
    iterations::Int
    epocsperiter::Float64 #Array{Float64}
    gradsperiter::Float64
    times::Array{Float64}
    fs::Array{Float64}   # recorded function values
    testerrors::Array{Float64}
    xfinal::Array{Float64}  #The final x of the method
    fail::AbstractString
    stepsize_multiplier::Float64
end

#Functions for getting and manipulating data
include("dataLoad.jl")
include("data_generation.jl")
include("logistic_defintions.jl") # includes all functions and definitions pertaining to logistic regression
include("load_ridge_regression.jl")
#Including method wrappers
include("minimizeFunc.jl") # minimizing a given prob::Prob using a method
include("minimizeFunc_grid_stepsize.jl")  # minimizing a given prob::Prob using a method and determines the stepsize using a grid search
include("boot_method.jl")
#Including test and problem generating functions
include("testing.jl")
#Including iterative methods for calculating search direction
allmethods = ["free_SVRG_nice", "SVRG_nice", "SAGA_nice", "SPIN", "SAGA", "SVRG", "SVRG2",  "2D", "2Dsec", "CMcoord", "CMgauss", "CMprev", "AMgauss","AMprev", "AMcoord", "BFGS", "BFGS_accel", "grad"];
recentmethods = ["free_SVRG_nice", "SVRG_nice", "SAGA_nice"]
for method in allmethods
    if method in recentmethods
        include(string("boot_", method , "!.jl")) # boot is a mutating function
        include(string("descent_", method , "!.jl")) # descent is a mutating function
    else
        include(string("boot_", method , ".jl"))
        include(string("descent_", method , ".jl"))
    end
end
include("descent_SAGApartition.jl")

#Including utilities, plotting, data analysis
include("plot_outputs_Plots.jl")
include("plot_SAGA_nice_Plots.jl")
include("get_saved_stepsize.jl");
include("load_fsol.jl");
include("../util/matrix_scaling.jl");
include("../util/matrix_rotation.jl");
include("../util/preprocessing.jl");
include("../util/power_iteration.jl");

#Additional
include("BFGS_update!.jl");
include("calculate_SAGA_rates_and_complexities.jl");
include("free_SVRG_settings.jl");
include("get_saved_stepsize.jl");
# include("../tmp/parallel_minimizeFunc_grid_stepsize.jl")